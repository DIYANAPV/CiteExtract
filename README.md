# CheckCite

**Citation verification for scientific papers, with evidence.**

CheckCite takes a paper — PDF, LaTeX, BibTeX, or plain text — and returns,
for every reference, two independent verdicts: a metadata-driven
top-level verdict (`VALID`, `FABRICATED`, `UNVERIFIABLE`) plus a
claim-support dimension (`SUPPORTED`, `CONTRADICTS`, `NEUTRAL`,
`UNVERIFIABLE`) telling whether the cited paper actually supports the
citing sentence. The two travel together with the passage that grounds
each finding. The design closes two gaps in prior work: existence-focused
tools do not check whether the cited paper actually supports the claim,
and semantic-alignment tools assume the reference has already been
resolved and retrieved. CheckCite does both in a single pass and returns
the evidence a human would need to judge borderline cases.

The semantic stage uses a multi-query retrieval method: citing
sentences are decomposed into sub-claims, evidence is retrieved for each
sub-claim, and the unioned passage set is scored against the full
original claim.

---

## Quick start

```bash
# 1. Clone and install
git clone <anonymized-repository-url>
cd checkcitation
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 2. Configure API keys
cp .env.example .env
# Edit .env — at minimum set OPENAI_API_KEY for semantic verification.

# 3. For PDFs, start GROBID (one-time)
docker run -d --name grobid -p 8070:8070 grobid/grobid:0.8.2-crf

# 4. Try it
python -m src verify paper.pdf --agentic   # full pipeline
python app.py                                # or use the web UI at localhost:7860
```

## Docker

A ready-to-run image will be published to a container registry. Build
locally in the meantime:

```bash
docker compose up --build        # web UI at http://localhost:7860
```

## How it works

CheckCite processes each reference through three layers before producing
a combined verdict.

```
Input: PDF / LaTeX / BibTeX / text
    │
    ▼
L1 — Existence cascade  (deterministic, no LLM)
    CrossRef (DOI authority) → Semantic Scholar → OpenAlex → PubMed
    + arXiv-to-publication bridge, dead-DOI detection, author cross-validation
    │  ┌── not found anywhere ──► FABRICATED
    ▼
L2 — Metadata validation (deterministic, no LLM)
    Field-by-field fuzzy comparison: title, authors, venue, year
    Chimera detection: title matches but author/venue/year wrong → flagged
    │
    ▼
L3 — Semantic verification (2 LLM calls per reference)
    Chunk cited paper (~512 chars) → decompose claim → retrieve per sub-claim
    → union + dedupe → verify full claim against the unioned passage set
    → SUPPORTED / NOT_SUPPORTED + evidence quote
    │
    ▼
Combined per-reference verdict (+ evidence passages + flags)
```

Implementation of the multi-query decomposition lives in
[`src/verification/multiquery.py`](src/verification/multiquery.py).

## Verification modes

| Mode | What it checks | LLM cost | When to use |
|------|----------------|----------|-------------|
| `--quick` | L1 + L2 only, rule-based | Free | "Are my references real?" |
| `--agentic` | L1 + L2 + multi-query L3 with triage (recommended) | ~$0.05/paper | Best accuracy on real papers |
| `comprehend` | L1 + passage retrieval only, no LLM judgement | Free | "What do my cited papers actually say?" |

## Web UI

```bash
python app.py      # http://localhost:7860
```

Upload a PDF and see one card per reference with the verdict,
bibliographic record, and the top evidence passage from the cited paper.
Every card has an inline **Agree / Disagree / Need info** review row
(FR9) so you can audit the tool's output; the top bar tracks review
progress and lets you export your decisions as CSV.

## Supported input formats

| Format | Notes |
|--------|-------|
| `.pdf` | Full analysis. Requires GROBID Docker. |
| `.tex` | References from companion `.bib` + citing contexts from body. |
| `.bib` | References only; forces `--quick` mode (no citing context). |
| `.txt` | Best-effort parsing. Quality depends on formatting. |

Limits: 50 MB, 500 references per paper.

## Configuration

| File | Purpose | Committed? |
|------|---------|-----------|
| `.env` | API keys and user-specific secrets | No |
| `config/config.yaml` | Pipeline knobs (thresholds, models, retrieval) | Yes |

Minimum `.env`:

```bash
OPENAI_API_KEY=sk-...          # required for --agentic
S2_API_KEY=...                  # optional but recommended (10× rate limit)
CROSSREF_MAILTO=you@uni.edu     # optional polite-pool email
OPENALEX_MAILTO=you@uni.edu     # optional polite-pool email
```

`--quick` mode works without any API key.

## Experiments

Experimental code lives under [`experiment/`](experiment/) and imports
the same primitives used in production code under `src/`. See
[`experiment/README.md`](experiment/README.md) for runners and
configurations.

## Output

Verdicts are written to `data/output/` as JSON. Each reference gets:

- A metadata verdict (`VALID` / `FABRICATED` / `UNVERIFIABLE`) and, in agentic mode, a claim verdict (`SUPPORTED` / `CONTRADICTS` / `NEUTRAL` / `UNVERIFIABLE`)
- A natural-language explanation
- Flags for human review
- Evidence trail: databases checked, metadata comparison
- Retrieved passages with their relevance scores (Agentic mode)
- A suggested action (remove citation, verify claim, no action)
- For `VALID` references in Agentic mode: corrected APA and BibTeX
  generated from the matched database record.

The web UI additionally offers:

- **Annotated PDF** (PDF uploads only) — the original paper with every
  detected citation marker highlighted in color by verdict, plus click-to-expand
  sticky notes carrying the verdict + explanation.
- **Problematic-refs BibTeX** — `.bib` file containing only flagged references,
  with an audit trail in each entry's `note` field.
- **Batch mode** — upload multiple papers or a `.zip`; get one aggregate
  dashboard, a per-paper rollup table, and CSV / JSON / BibTeX exports.

## Project structure

```
checkcitation/
├── app.py                        Web UI (Gradio)
├── config/config.yaml            Pipeline settings
├── src/                          Production code
│   ├── pipeline.py               Main orchestrator
│   ├── parsers/                  L1: BibTeX / LaTeX / text / GROBID
│   ├── citation/                 Citation detection + context extraction
│   ├── verification/
│   │   ├── existence.py          L2 cascade: CrossRef → S2 → OpenAlex → PubMed
│   │   ├── metadata.py           L3 field-level validation + chimera detection
│   │   ├── multiquery.py         Decomposition + multi-query retrieval
│   │   ├── comprehension.py      BM25 + dense + RRF + FlashRank
│   │   ├── agentic/              Triage + MetadataAgent + ClaimAgent
│   │   └── api_clients/          CrossRef, S2, OpenAlex, PubMed, Unpaywall, LLM
│   ├── classification/           Final verdict decision tree
│   └── models/                   Pydantic data models
├── experiment/                   Paper experiments (imports from src/)
│   ├── ours/                     Main runner + ablations
│   ├── baselines/                Comparison baselines
│   └── results/                  JSON outputs
├── paper/                        LaTeX sources, tables, figures
└── tests/                        Unit + integration tests
```

## Running tests

```bash
python -m pytest tests/ -v
python -m pytest tests/ -v --ignore=tests/test_existence.py   # offline
```

## Citing CheckCite

```bibtex
@misc{checkcite,
  title  = {CheckCite},
  author = {Anonymous},
  year   = {2026},
  note   = {Submitted for double-blind review}
}
```

## Licence

Released under a permissive open-source licence; see `LICENSE`.
