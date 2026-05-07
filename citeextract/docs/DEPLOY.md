# Deployment Guide

This app can be tested locally today and deployed to Cloud Run later with no code changes. The cloud deploy section is a stub that expands when you are ready to run it.

---

## 1. Environment variables

Set these in `.env` (local) or via your host's secret manager (cloud).

| Variable | Required | Purpose |
| --- | --- | --- |
| `OPENAI_API_KEY` | Required for Agentic mode | LLM calls during claim verification |
| `REVIEW_ACCESS_TOKEN` | Recommended for public URLs | Unguessable string; anyone without it sees a 403 page |
| `MONTHLY_BUDGET_USD` | Recommended for public URLs | Hard ceiling on cumulative OpenAI spend per calendar month. New analyses are refused once reached. Unset = no cap. See §5 *Spend ceiling*. |
| `DAILY_ANALYSIS_LIMIT` | Optional | Global papers/day across all users, counts singles + batch papers (default: 40) |
| `HOURLY_IP_LIMIT` | Optional | Single-paper analyses/hour per IP (default: 10) |
| `HOURLY_IP_BATCH_LIMIT` | Optional | Batch submissions/hour per IP (default: 2) |
| `BATCH_MAX_PAPERS_PER_BATCH` | Optional | Papers allowed in one batch submission (default: 30) |
| `PORT` | Optional | HTTP port (default: 7860) |
| `S2_API_KEY` | Optional | Lifts Semantic Scholar's 1 RPS free-tier limit |
| `GROBID_SERVICE_URL` | Auto in docker-compose | URL of the GROBID service |

## HTTP routes

| Path | Purpose | Gate |
| --- | --- | --- |
| `/` | Gradio web UI | token required (if `REVIEW_ACCESS_TOKEN` set) |
| `/health` | Liveness probe, plain `ok` | public |
| `/docs` | Swagger UI — programmatic API spec | public |
| `/redoc` | Alternative API docs | public |
| `/openapi.json` | Machine-readable OpenAPI schema | public |

The spec paths are intentionally public so that developers can discover the
API surface without the access token. Actual API calls still require the
token in `?token=` or as the `citeextract_token` cookie.

Leaving `REVIEW_ACCESS_TOKEN` unset disables the gate — fine for local dev, **do not do this on a public URL**.

---

## 2. Set the OpenAI project-level spending cap ($50/month)

Most important step before any public deploy. This is your hard ceiling — even if every other safety net fails, OpenAI refuses the request that would put you over.

1. Go to https://platform.openai.com/settings/organization/projects
2. Click **Create project** — name it something like `citeextract-review`
3. Inside the project, open **Limits**
4. Set **Monthly budget** to `$50` (or whatever cap you're comfortable with)
5. Set **Email alerts** at `$10`, `$25`, `$40` so you're warned early
6. Under **API keys**, create a key *scoped to this project*
7. Put that key in `.env` as `OPENAI_API_KEY=sk-...`

Do NOT reuse your university/personal key. A project-scoped key with its own limit means a leak can't drain your main quota.

---

## 3. Local test checklist

Before pointing any reviewer at this instance, confirm each of these locally:

```bash
# 1. Install
pip install -e ".[dev]"
python -m spacy download en_core_web_sm

# 2. Run unit + integration tests
pytest                            # unit tests
pytest tests/test_integration.py  # end-to-end on clean_paper.tex

# 3. Launch with the gate enabled
REVIEW_ACCESS_TOKEN=testtoken python -m citeextract_ui
```

Then in a browser, verify:

- `http://localhost:7860/` → shows the gate page (403)
- `http://localhost:7860/?token=testtoken` → loads the UI and sets a cookie
- `http://localhost:7860/health` → returns `ok` without the token
- Upload `data/test_inputs/clean_paper.tex` with only *Existence & Metadata* checked → completes in a few seconds, 4 verdicts
- The progress bar actually moves as stages complete
- Try uploading a file > 10 MB → friendly error, no crash
- Try uploading a `.pdf` that's actually a text file → friendly error

---

## 4. Sharing the link with reviewers

For a conference submission under double-blind review, put the token in the paper itself:

> *An interactive demo is available at* `https://example.run.app/?token=R3V1eW2026`

Reviewers click once; the cookie keeps them logged in for 30 days. Rotate the token after acceptance/camera-ready so the link in the pre-review PDF stops working.

The token only needs to resist casual scraping and bots — not determined attackers. The real safety nets are the OpenAI cap (§2) and the rate limits (§1).

---

## 5. Cloud Run deploy

`scripts/deploy_gcp.sh` is the source of truth — it creates the project, links billing, sets a budget alert, stores secrets in Secret Manager, deploys GROBID, builds and deploys the main app, and prints the reviewer link. Re-running it is safe.

### One-time prerequisites

1. Activate the $300 / 90-day free trial at https://console.cloud.google.com (credit card required, but the account *pauses* at $0 — it does not auto-charge)
2. Install the [gcloud CLI](https://cloud.google.com/sdk/docs/install), then `gcloud auth login`
3. Complete §2 above (OpenAI project-scoped key with monthly cap) — the script needs the key
4. `gcloud billing accounts list` and copy the `ACCOUNT_ID`

### Run

```bash
PROJECT_ID=citecheck-demo-2026 \
BILLING_ACCOUNT_ID=XXXXXX-XXXXXX-XXXXXX \
OPENAI_API_KEY=sk-proj-... \
REVIEW_ACCESS_TOKEN=$(python3 -c "import secrets; print(secrets.token_urlsafe(16))") \
./scripts/deploy_gcp.sh
```

Pick a `PROJECT_ID` that doesn't contain your name or institution — the project ID is occasionally visible (build logs, custom domains) and would break double-blind review.

The first run takes ~15–20 minutes, mostly the Cloud Build step on the torch/sentence-transformers image. Subsequent re-runs (config changes, code updates) take ~3–5 minutes.

### Deployed shape

- **GROBID** (`grobid/grobid:0.8.2-crf`, 4Gi/2vCPU): `min=0 max=2`, default request-based billing — pay only when a PDF is being parsed. Reviewers eat one ~30s cold start per session. HTTP startup probe at `/api/isalive` so the JVM finishes booting before traffic hits.
- **Main app** (this repo's [Dockerfile](../Dockerfile), 4Gi/1vCPU): `min=max=1`, default request-based billing, `concurrency=5`. One warm instance avoids cold starts for reviewers and sidesteps the multi-instance issues with the in-process job store ([citeextract/backend/job_store.py](../backend/job_store.py)) and on-disk SQLite cache (`data/cache/api_cache.db`). HTTP startup probe at `/health`.
- **Secrets** `openai-api-key` and `review-access-token` in Secret Manager, mounted as env vars on the runtime service account `citeextract-sa@<PROJECT>.iam.gserviceaccount.com`.
- **Budget alert** at 25 / 50 / 90 / 100 % of $200 (override with `BUDGET_USD=`).
- **Smoke test**: the script `curl`s `/health` after deploy and exits non-zero if it doesn't return 200.

### Spend ceiling

Independent of any OpenAI org-level cap (which a `member` account can't set), the deploy bakes in a per-month USD ceiling enforced in code at [citeextract/backend/verification/spend_guard.py](../backend/verification/spend_guard.py).

- Every LLM call's USD cost is appended to `data/cache/spend/<YYYY-MM>.json` via an `fcntl.flock`-guarded write (one record per call, atomic).
- Before each analysis, the rate-limit gate in [citeextract/backend/rate_limit.py](../backend/rate_limit.py) calls `spend_guard.check_budget()`. If cumulative spend has reached `MONTHLY_BUDGET_USD`, reviewers see a "service paused for the month" message instead of starting an analysis. In-flight analyses are not killed — the worst overrun is `<concurrency> × max_cost_per_paper`, ~$2.50 for the default config.
- Default in the deploy script: `MONTHLY_BUDGET_USD=150`. Override at deploy time: `MONTHLY_BUDGET_USD=50 ./scripts/deploy_gcp.sh`.
- Auto-rolls over on calendar month (UTC).
- Counter survives across requests on the warm `min=1` instance. If Cloud Run evicts the instance (rare), the file is lost and the next instance starts at $0 — the OpenAI dashboard remains your second pair of eyes.
- Bump or remove the cap without redeploying: `gcloud run services update citeextract --region=us-central1 --update-env-vars=MONTHLY_BUDGET_USD=300`.

### Cost expectation

After the per-billing-account free tier (180k vCPU-s, 360k GiB-s, 2M requests / month), this configuration runs ≈ **$45 over 90 days** for typical peer-review traffic — roughly $5/mo compute on the main service, $5/mo for sporadic GROBID use, $5/mo for build/storage/egress. The $300 trial credit is the outer bound; the $200 budget alert is the inner one.

If you'll drive the async API endpoint hard (long-running pipeline jobs through `POST /api/v1/jobs`), pass `MAIN_CPU_ALWAYS=1` to the script. That switches the main service to instance-based billing (`--no-cpu-throttling`) so FastAPI BackgroundTasks aren't CPU-throttled between requests. Cost rises to ≈$90/mo, ~$270 over 90 days — still inside the credit, but tighter.

### Known limitation: instance-bounded state

The job store, API cache, and verdict cache all live on the container's local filesystem. With `min=max=1` this is fine — the same instance handles every request and survives across deploys (Cloud Run keeps the old revision serving until the new one is healthy). It would break under autoscaling. If you ever need to lift `max-instances`, migrate [citeextract/backend/job_store.py](../backend/job_store.py) and [citeextract/backend/verification/cache.py](../backend/verification/cache.py) to Firestore + GCS first.

### Rotate the token after camera-ready

```bash
printf "%s" "new-token" | gcloud secrets versions add review-access-token --data-file=-
gcloud run services update citeextract --region=us-central1 \
  --update-secrets="REVIEW_ACCESS_TOKEN=review-access-token:latest"
```

The pre-review PDF's link stops working immediately. (`printf` instead of `echo -n` because some shells add a newline anyway, which Secret Manager stores as part of the secret value.)

### Wind down after publication

```bash
gcloud projects delete <PROJECT_ID>
```

Marks the project for deletion in 30 days, taking the services, secrets, build artifacts, and budget with it. Cancel within 30 days if you change your mind.

---

## 6. What to watch once it's live

- **OpenAI dashboard** — usage graph for the `citeextract-review` project
- **GCP budget alerts** email
- **Container logs** — search for `HTTP 403` (gate rejections), `Hourly limit reached`, `Daily analysis limit` to see if you're being probed
- **The app itself** — click through it once a week to make sure it still works
