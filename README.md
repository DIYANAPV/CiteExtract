# CheckCitation

Open-source citation hallucination detection system. Give it a paper (PDF, LaTeX, BibTeX, or plain text), get a detailed report on every citation: fabricated or misrepresented.

## What it detects

| Failure Mode | Description | How |
|-------------|-------------|-----|
| **Fabricated** | Citation doesn't exist in any scholarly database, has been retracted, or has incorrect metadata | Existence check + metadata validation |
| **Misrepresented** | Real paper, but retrieved passages contradict the claim made about it | Full-text passage retrieval + LLM analysis |

## Quick start

```bash
# 1. Clone and install
cd checkcitation
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 2. Set up your environment
cp .env.example .env
# Edit .env and add your API keys (see "Configuration" below)

# 3. For PDFs: start GROBID (one-time, stays running)
docker run -d --name grobid -p 8070:8070 grobid/grobid:0.8.2-crf

# 4. Verify a paper
python -m src verify paper.pdf --quick
python -m src verify refs.bib --quick
python -m src verify paper.pdf --standard

# 5. Or use the web UI
python app.py    # opens at http://localhost:7860

# 6. Check output
ls data/output/    # JSON reports
```

## Web UI

Launch the browser-based interface:

```bash
python app.py
```

Opens at `http://localhost:7860` with an Analyze tab where you can:
- Upload PDF/LaTeX/BibTeX/text files
- Select analysis options: Existence & Metadata, Claim Verification
- Choose mode: Standard (pipeline) or Agentic (smart triage + focused agents)
- Upload reference PDFs for non-open-access papers
- See color-coded results with metadata comparisons, passages, claim verdicts, and corrected citations

## How to use (CLI)

```bash
# Quick mode (free, no LLM) — checks if references exist + metadata matches
python -m src verify /path/to/paper.pdf --quick

# Standard mode (needs OPENAI_API_KEY) — retrieves passages + LLM claim verification
python -m src verify /path/to/paper.pdf --standard

# Agentic mode (needs OPENAI_API_KEY) — smart triage + focused agents for ambiguous cases
python -m src verify /path/to/paper.pdf --agentic

# Retry references that previously failed (clears NOT_FOUND cache)
python -m src verify /path/to/paper.pdf --retry-failed

# Human-readable table output (instead of JSON)
python -m src verify /path/to/paper.pdf --format table

# Parse only (no verification, just extract references)
python -m src parse /path/to/paper.pdf

# Comprehension mode — retrieve relevant passages from cited papers
python -m src comprehend /path/to/paper.pdf

# With user-provided PDFs for paywalled references
python -m src comprehend /path/to/paper.pdf --ref-pdfs /path/to/reference-pdfs/
```

## Verification modes

| Mode | What it checks | LLM cost | When to use |
|------|---------------|----------|-------------|
| **Quick** (`--quick`) | Existence + metadata, rule-based | Free | "Are my references real?" |
| **Standard** (`--standard`) | Quick + passage retrieval + LLM claim verification | ~$0.01-0.05/paper | "Do the cited papers support my claims?" |
| **Agentic** (`--agentic`) | Smart triage + focused agents for ambiguous cases | ~$0.05-0.10/paper | Best accuracy, handles edge cases |
| **Comprehend** (`comprehend`) | Retrieve relevant passages from cited papers | Free | "What do my cited papers actually say?" |

### Quick vs. Standard vs. Agentic

**Quick** uses configurable similarity thresholds and a deterministic cascade through databases. Fast, cheap, transparent. Handles common edge cases automatically:
- Truncated author lists (5 of 50 authors → detected as valid truncation, not mismatch)
- Preprint-vs-publication differences (arXiv 2021 → CVPR 2022 → upgrades to published version via CrossRef)
- Venue name variations (NeurIPS vs. "Advances in Neural Information Processing Systems")
- Dead DOI detection (HTTP HEAD to doi.org — catches registered but unresolvable DOIs)
- Author cross-validation (checks authors against a second database, flags suspects found in zero DBs)

**Standard** adds claim verification: fetches full text of each cited paper, retrieves the most relevant passages using BM25+dense hybrid retrieval with FlashRank neural reranking, then asks an LLM to judge whether the citing sentence accurately represents the cited paper. Citation markers are stripped from retrieval queries for cleaner matching.

**Agentic** uses the same shared L2+L3 pipeline as Quick, then applies smart triage: clear-cut cases (exact matches, obvious fabrications) are resolved by rules without any LLM calls. Only ambiguous references get dispatched to focused agents:
- **MetadataAgent** — investigates metadata discrepancies with format-aware comparison
- **ClaimAgent** — verifies claims against retrieved passages
- This means most references are resolved for free; LLM costs only apply to the ~10-30% that need deeper investigation
- VALID verdicts include corrected APA and BibTeX citations generated from database metadata

## Input limits

- **Max file size:** 50 MB
- **Max references:** 500 per paper

## Supported input formats

| Format | What you get |
|--------|-------------|
| **.pdf** | Full analysis. Requires GROBID Docker (see below). All modes. |
| **.tex** | References from companion .bib + citing contexts from body. All modes. |
| **.bib** | References only. Quick mode forced (no citing context). |
| **.txt** | Best-effort parsing. Quality depends on formatting. |

## Configuration

| File | What goes here | Committed to git? |
|------|---------------|-------------------|
| **`.env`** | API keys and user-specific settings (secrets) | No (in .gitignore) |
| **`config/config.yaml`** | Application behavior (thresholds, timeouts, models) | Yes |

### Setting up `.env`

```bash
cp .env.example .env
```

Then edit `.env`:

```bash
# Required for Standard and Agentic modes
OPENAI_API_KEY=sk-your-key-here

# Strongly recommended — increases Semantic Scholar rate limit from 1 to 10 req/s
S2_API_KEY=your-key-here

# Polite pool emails — higher API rate limits
CROSSREF_MAILTO=your.email@university.edu
OPENALEX_MAILTO=your.email@university.edu
```

**Quick mode works without any API keys.** But without `S2_API_KEY`, rate limits may cause real papers to be misclassified as FABRICATED.

## PDF parsing

PDF input requires GROBID running as a Docker container:

```bash
docker run -d --name grobid -p 8070:8070 grobid/grobid:0.8.2-crf
curl http://localhost:8070/api/isalive    # Should return "true"
```

## How it works

### Pipeline (Quick/Standard modes)

```
Input file (.bib / .tex / .pdf / .txt)
    |
    v
L1: PARSE -- Extract references + citing contexts
    |
    v
L2: EXISTENCE CHECK -- DOI->CrossRef, title->S2->OpenAlex->PubMed
    |  + Web fallback: URL resolution + Wayback Machine
    |  + Preprint upgrade: if DB returns arXiv but ref cites conference,
    |    automatically tries CrossRef/OpenAlex for the published version
    |  + Dead DOI detection (HTTP HEAD to doi.org)
    |  + Author cross-validation (query second DB, flag suspect authors)
    |
    v
L3: METADATA VALIDATION -- Field-by-field comparison
    |  + Author truncation detection (subset check, not just Jaccard)
    |  + Preprint-vs-publication venue detection (CLOSE_MATCH, not MISMATCH)
    |
    v
CLAIM VERIFICATION (Standard mode only):
    |  Strip citation markers from query -> chunk (512 chars, 50-char overlap)
    |  -> BM25+dense retrieval -> RRF fusion -> FlashRank neural reranker -> top 3
    |  -> LLM analysis (temperature 0.0 for deterministic output)
    |
    v
L5: CLASSIFICATION -- Decision tree -> verdict per reference
    |  + Corrected APA + BibTeX from database metadata for VALID verdicts
    |
    v
L6: REPORT -- JSON with evidence trail + corrected citations
```

### Agentic pipeline

```
Input file
    |
    v
L1: PARSE (same as above)
    |
    v
L2: EXISTENCE CHECK (shared with Quick/Standard)
    |  + Dead DOI detection (HTTP HEAD to doi.org)
    |  + Author cross-validation (query second DB, flag suspects)
    |
    v
L3: METADATA VALIDATION (shared with Quick/Standard)
    |
    v
TRIAGE -- classify each reference:
    |  Clear-cut (exact match, obvious fabrication) → resolve by rules (no LLM)
    |  Ambiguous → dispatch focused agent:
    |
    +-- MetadataAgent: investigates metadata discrepancies
    |   (format-aware author comparison, word-level title diff)
    +-- ClaimAgent: verifies claims against retrieved passages
    |
    v
L6: REPORT -- JSON with evidence trail + corrected citations
```

### Caching

- **GROBID XML cache** (`data/cache/grobid_{hash}.xml`)
- **ParsedPaper cache** (`data/cache/parsed_{hash}.json`)
- **API response cache** (`data/cache/api_cache.db`) — SQLite, auto-expiring

Re-running the same paper is near-instant. To retry failed references: `--retry-failed`.

## Output

Reports are saved to `data/output/`. Each citation gets:
- A verdict (FABRICATED, MISREPRESENTED, VALID, or UNVERIFIABLE)
- An explanation of why
- Flags for human review
- Evidence trail (databases checked, metadata comparison)
- Retrieved passages with claim verdicts (Standard/Agentic modes)
- Suggested action (remove citation, verify claim, or no action)

## Project structure

```
checkcitation/
+-- app.py                        # Web UI (Gradio)
+-- .env.example                  # Template for API keys
+-- config/config.yaml            # Application settings
+-- src/
|   +-- config.py                 # Centralized config loader
|   +-- pipeline.py               # Main orchestrator
|   +-- __main__.py               # CLI entry point
|   +-- parsers/                  # L1: BibTeX, LaTeX, text, GROBID (PDF)
|   +-- citation/                 # Citation detection + context extraction + format detection
|   +-- verification/
|   |   +-- existence.py          # L2: Database cascade + web fallback + dead DOI + author cross-validation
|   |   +-- metadata.py           # L3: Field validation
|   |   +-- comprehension.py      # Passage retrieval (BM25 + dense + RRF + FlashRank reranker)
|   |   +-- matching.py           # Fuzzy matching utilities (title, author, year, venue)
|   |   +-- filters.py            # Citation filtering utilities
|   |   +-- api_clients/          # CrossRef, S2, OpenAlex, PubMed, Unpaywall, LLM
|   |   +-- agentic/              # Triage + focused agents (MetadataAgent, ClaimAgent)
|   +-- classification/           # L5: Decision tree + corrected citation output
|   +-- report/                   # L6: JSON generation
|   +-- models/                   # Pydantic data models
+-- scripts/                      # Benchmark and evaluation scripts
+-- tests/                        # Unit + integration tests
+-- data/
|   +-- cache/                    # API response cache (auto-created)
|   +-- output/                   # Generated reports (auto-created)
+-- notes/                        # Design docs and research notes
+-- requirements.txt
```

## Running tests

```bash
python -m pytest tests/ -v
python -m pytest tests/ -v --ignore=tests/test_existence.py  # no network
```

## Evaluation

### Benchmark A — Fabrication detection

```bash
python scripts/benchmark.py
```

### Retrieval evaluation

```bash
python scripts/eval_retrieval.py   # BM25 vs Dense vs Hybrid
python scripts/eval_rrf_k.py       # RRF k parameter sweep
```
