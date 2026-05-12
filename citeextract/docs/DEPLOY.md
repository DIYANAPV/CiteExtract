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
