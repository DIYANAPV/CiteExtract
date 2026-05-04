# Paper artifacts — CiteExtract semantic + metadata benchmarks

Reproduction package for the two benchmark tables. Self-contained: data, prompts, runners, and results all live under this folder.

## Layout

```
experiment/paper/
├── data/          benchmark JSONLs (semantic 741, metadata 302)
├── prompts/       three system prompts (semantic, metadata, metadata+search)
├── semantic/      runner + LaTeX aggregator for the semantic table
├── metadata/      runner + LaTeX aggregator for the metadata table
├── openweight/    transformers-based runner for open-weight rows + SLURM script
└── results/       per-cell CSVs, partial logs, LaTeX tables, JSON metric breakdowns
```

## Headline results

### Semantic table — 741 instances, 4 models × 3 conditions

| Model | Title only | + Abstract | + Passages |
|---|---|---|---|
| gpt-4o-mini | 72.74 | 71.39 | **83.81** ← production |
| gpt-4o | 80.43 | 78.00 | 85.02 |
| Qwen3-8B | 76.11 | 71.79 | 83.00 |
| Llama-3.1-8B-Instruct | 56.55 | 52.36 | 53.98 |

### Metadata table — 302 instances (151 valid / 151 fabricated)

| Method | Acc | F1 (Fab) |
|---|---|---|
| gpt-4o-mini | 58.94 | 59.48 |
| gpt-4o-mini + web search | 82.78 | 81.43 |
| gpt-4o | 61.59 | 63.06 |
| gpt-4o + web search | 84.77 | 85.16 |
| Qwen3-8B | 43.38 | 37.36 |
| Llama-3.1-8B-Instruct | 45.03 | 20.95 |
| **CiteExtract pipeline (ours)** | **96.36** | **96.37** |

Full per-cell metrics (P/R/F1, time, cost): [`results/tab_semantic_results_full.json`](results/tab_semantic_results_full.json), [`results/tab_metadata_results_full.json`](results/tab_metadata_results_full.json).

## Reproducing

OpenAI rows need `OPENAI_API_KEY` in `.env`. Open-weight rows need a HuggingFace login for gated models like Llama (`huggingface-cli login`).

### Semantic table

```bash
# OpenAI rows (gpt-4o-mini, gpt-4o)
python -m experiment.paper.semantic.runner --full

# Open-weight rows (run on a GPU box)
python -m experiment.paper.openweight.runner --model Qwen/Qwen3-8B --task semantic --full
python -m experiment.paper.openweight.runner --model meta-llama/Llama-3.1-8B-Instruct --task semantic --full

# Aggregate to LaTeX
python -m experiment.paper.semantic.make_table
```

### Metadata table

```bash
# OpenAI rows + the deterministic CiteExtract row (uses src/verification/*)
python -m experiment.paper.metadata.runner --full

# Open-weight rows
python -m experiment.paper.openweight.runner --model Qwen/Qwen3-8B --task metadata --full
python -m experiment.paper.openweight.runner --model meta-llama/Llama-3.1-8B-Instruct --task metadata --full

# Aggregate
python -m experiment.paper.metadata.make_table
```

### Open-weight rows on a SLURM cluster

```bash
sbatch experiment/paper/openweight/run.sh Qwen/Qwen3-8B
sbatch experiment/paper/openweight/run.sh meta-llama/Llama-3.1-8B-Instruct
```

The script runs both tasks (semantic + metadata) for one model. Output lands in [`results/`](results/) directly. Wall time ~1.5–3 hours per model on an L40S/A3090.

## Notes

- **Resume:** every cell writes a `*_partial_*.jsonl` log. Re-running the same command picks up where it left off; pass `--no-resume` to start fresh.
- **Determinism:** OpenAI cells use `temperature=0` + JSON mode. Open-weight cells use greedy generation. Both are deterministic on re-run.
- **Production cell** (the bolded row in each table) is gpt-4o-mini × `title_abstract_passages` for semantic, and the deterministic CiteExtract cascade (`metadata/run_production.py`) for metadata.
