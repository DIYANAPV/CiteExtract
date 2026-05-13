# CiteExtract

**Citation verification for scientific papers, with evidence.**

CiteExtract takes a paper — PDF, LaTeX, BibTeX, or plain text — and for every reference returns two independent verdicts: whether the cited paper exists with the stated bibliographic details, and whether it actually supports the claim made for it. Each verdict comes with the passage that grounds it.

## Install

```bash
cd citeextract
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m spacy download en_core_web_sm
```

For PDF input, run GROBID once (Docker):

```bash
docker run -d --name grobid -p 8070:8070 grobid/grobid:0.8.2-crf
```

## Run

```bash
python -m citeextract verify paper.pdf --agentic     # CLI
python -m citeextract_ui                             # web UI at http://localhost:7860
uvicorn citeextract.api:app --port 8000              # HTTP API
```

Or the full stack via Docker:

```bash
docker compose up --build
```

## What you get

For every reference, two independent verdicts plus the evidence behind them:

- **Metadata verdict** — `VALID`, `FABRICATED`, `UNVERIFIABLE`
- **Claim-support verdict** — `SUPPORTED`, `CONTRADICTS`, `NEUTRAL`, `UNVERIFIABLE`




## Inputs

`.pdf` (needs GROBID), `.tex` (with `.bib` companion), `.bib` (forces `--quick`), `.txt` (best-effort). Limits: 50 MB, 500 references per paper.

## Configuration

Set in `.env`:

```bash
OPENAI_API_KEY=your-openai-key  # required for --agentic
S2_API_KEY=your-s2-key          # optional
CROSSREF_MAILTO=you@uni.edu     # optional polite-pool
OPENALEX_MAILTO=you@uni.edu     # optional polite-pool
```

Pipeline thresholds and model choices live in [`config/config.yaml`](config/config.yaml).



## Project layout

```
citeextract/
├── backend/                  import citeextract.X     
├── frontend/                 import citeextract_ui.X  
├── tests/, config/, docs/, scripts/, assets/
└── pyproject.toml, Dockerfile, docker-compose.yml, .env.example
```

## Documentation

- [`docs/DEPLOY.md`](docs/DEPLOY.md) — self-hosted deployment (Cloud Run, OpenAI spend cap, token gate, rate limits)


