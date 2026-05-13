# Deployment Guide

For local use. Set up `.env` (see `.env.example`), then run with Docker or natively.

## Environment variables

| Variable | Required | Purpose |
| --- | --- | --- |
| `OPENAI_API_KEY` | Required for Agentic mode | LLM calls for claim verification |
| `REVIEW_ACCESS_TOKEN` | Recommended for public URLs | Unguessable string; without it, callers hit a 403 page |
| `MONTHLY_BUDGET_USD` | Recommended for public URLs | Hard ceiling on OpenAI spend per calendar month |
| `S2_API_KEY` | Optional | Lifts Semantic Scholar's free-tier rate limit |
| `GROBID_SERVICE_URL` | Auto in docker-compose | URL of the GROBID service |
| `PORT` | Optional | HTTP port (default: 7860) |

## HTTP routes

| Path | Purpose | Gate |
| --- | --- | --- |
| `/` | Web UI | token required (if `REVIEW_ACCESS_TOKEN` set) |
| `/health` | Liveness probe | public |
| `/docs` | OpenAPI / Swagger UI | public |
| `/openapi.json` | OpenAPI schema | public |

Leaving `REVIEW_ACCESS_TOKEN` unset disables the gate. Fine locally; do not do this on a public URL.

## OpenAI spending cap

Recommended before exposing the app on any public URL.

1. https://platform.openai.com/settings/organization/projects → create a project for this app
2. Project → **Limits** → set a Monthly budget
3. Project → **API keys** → create a key scoped to the project
4. Put it in `.env` as `OPENAI_API_KEY=your-openai-key`

Use a project-scoped key, not your personal one.

## Local test

```bash
pip install -e ".[dev]"
python -m spacy download en_core_web_sm
pytest

REVIEW_ACCESS_TOKEN=testtoken python -m citeextract_ui
```

Then open `http://localhost:7860/?token=testtoken`.

## Cloud Run deploy

`scripts/deploy_gcp.sh` creates the project, links billing, stores secrets, deploys GROBID + the main app, and prints the reviewer URL. Re-running is safe.

### Prerequisites
1. Activate the $300 / 90-day free trial at https://console.cloud.google.com
2. Install [gcloud CLI](https://cloud.google.com/sdk/docs/install), then `gcloud auth login`
3. Create an OpenAI project-scoped key with a monthly cap (see above)
4. `gcloud billing accounts list` → copy the `ACCOUNT_ID`

### Run

```bash
PROJECT_ID=your-project-id \
BILLING_ACCOUNT_ID=XXXXXX-XXXXXX-XXXXXX \
OPENAI_API_KEY=your-openai-key \
REVIEW_ACCESS_TOKEN=$(python3 -c "import secrets; print(secrets.token_urlsafe(16))") \
./scripts/deploy_gcp.sh
```

Pick a `PROJECT_ID` that doesn't contain your name or institution — it shows up in build logs.

First run takes ~15–20 minutes (Cloud Build on the torch image). Re-runs take ~3–5.

The script prints the reviewer URL at the end: `https://<service>.run.app/?token=<token>`. Put that in the paper.

### Rotate the token after camera-ready

```bash
printf "%s" "<new-token>" | gcloud secrets versions add review-access-token --data-file=-
gcloud run services update citeextract --region=us-central1 \
  --update-secrets="REVIEW_ACCESS_TOKEN=review-access-token:latest"
```

### Wind down

```bash
gcloud projects delete <PROJECT_ID>
```
