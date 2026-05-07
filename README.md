# CiteExtract — research repository

Two folders:

- **[`citeextract/`](citeextract/)** — the product. Self-contained: pyproject, Dockerfile, tests, docs, config all live inside. See [`citeextract/README.md`](citeextract/README.md) for install + run.
- **[`experiment/paper/`](experiment/paper/)** — paper artefacts for the two benchmark tables: data, prompts, runners, results. See [`experiment/paper/README.md`](experiment/paper/README.md) for reproduction.

## Quick start

```bash
cd citeextract
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m spacy download en_core_web_sm
python -m citeextract_ui                 # web UI on http://localhost:7860
```
