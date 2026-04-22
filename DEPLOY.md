# Deployment Guide

This app can be tested locally today and deployed to Cloud Run later with no code changes. The cloud deploy section is a stub that expands when you are ready to run it.

---

## 1. Environment variables

Set these in `.env` (local) or via your host's secret manager (cloud).

| Variable | Required | Purpose |
| --- | --- | --- |
| `OPENAI_API_KEY` | Required for Agentic mode | LLM calls during claim verification |
| `REVIEW_ACCESS_TOKEN` | Recommended for public URLs | Unguessable string; anyone without it sees a 403 page |
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
token in `?token=` or as the `checkcitation_token` cookie.

Leaving `REVIEW_ACCESS_TOKEN` unset disables the gate — fine for local dev, **do not do this on a public URL**.

---

## 2. Set the OpenAI project-level spending cap ($50/month)

Most important step before any public deploy. This is your hard ceiling — even if every other safety net fails, OpenAI refuses the request that would put you over.

1. Go to https://platform.openai.com/settings/organization/projects
2. Click **Create project** — name it something like `checkcitation-review`
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
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 2. Run unit + integration tests
pytest tests/                  # unit tests
pytest tests/test_integration.py  # end-to-end on clean_paper.tex

# 3. Launch with the gate enabled
REVIEW_ACCESS_TOKEN=testtoken python app.py
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

## 5. Cloud Run deploy *(stub — fill in when ready)*

Intended shape:

1. `gcloud run deploy grobid` — separate service for GROBID
2. `gcloud run deploy checkcitation` — main app, points `GROBID_SERVICE_URL` at the GROBID service
3. Both in `us-central1` (always-free tier)
4. Put secrets (`OPENAI_API_KEY`, `REVIEW_ACCESS_TOKEN`) in Secret Manager; mount as env vars
5. Set a GCP budget alert at $50/$100/$200 (the $300 new-account trial is the outer bound)
6. `min-instances=1` during the active review period so reviewers don't hit cold starts

Write the actual commands when you're ready to run them.

---

## 6. What to watch once it's live

- **OpenAI dashboard** — usage graph for the `checkcitation-review` project
- **GCP budget alerts** email
- **Container logs** — search for `HTTP 403` (gate rejections), `Hourly limit reached`, `Daily analysis limit` to see if you're being probed
- **The app itself** — click through it once a week to make sure it still works
