# Paper artifacts — CiteExtract benchmarks

Reproduction package for the metadata, semantic, and prevalence studies. Self-contained: data, prompts, runners, and results all live under this folder.

## Layout

```
experiment/paper/
├── data/                benchmark JSONLs (semantic 741, metadata 302)
├── prompts/             three system prompts (semantic, metadata, metadata+search)
├── metadata/            metadata-task code (API + open-weight runners + LaTeX aggregator)
├── semantic/            semantic-task code (API + open-weight runners + LaTeX aggregator)
├── _shared/             shared open-weight infrastructure (TransformersClient, SLURM script)
├── analysis/            cross-task statistical analysis (bootstrap CIs, McNemar)
├── prevalence/          NeurIPS 2025 prevalence study (separate self-contained pipeline)
├── results/
│   ├── metadata/{csv,partial,tables,summaries}/
│   ├── semantic/{csv,partial,tables,summaries}/
│   └── significance/    cross-task significance JSON + Markdown summary
└── _trash/              items kept for safety; safe to remove later
```

Each task folder is symmetric:

```
metadata/                              semantic/
├── runner.py             (API)        ├── runner.py             (API)
├── runner_openweight.py  (HF local)   ├── runner_openweight.py  (HF local)
├── run_production.py     (CiteExtract └── (no production cell on this task)
│                          cascade)
├── llm_clients.py        (API client)
├── prompt_builder.py
└── make_table.py         (→ tables/)
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

Full per-cell metrics (P/R/F1, time, cost): [`results/semantic/tables/tab_semantic_results_full.json`](results/semantic/tables/tab_semantic_results_full.json), [`results/metadata/tables/tab_metadata_results_full.json`](results/metadata/tables/tab_metadata_results_full.json).

## Reproducing

OpenAI rows need `OPENAI_API_KEY` in `.env`. Open-weight rows need a HuggingFace login for gated models like Llama (`huggingface-cli login`) and the deps in [`_shared/requirements_openweight.txt`](_shared/requirements_openweight.txt).

### Semantic table

```bash
# OpenAI rows (gpt-4o-mini, gpt-4o, gpt-5*, gpt-5.5*)
python -m experiment.paper.semantic.runner --full

# Open-weight rows (run on a GPU box)
python -m experiment.paper.semantic.runner_openweight --model Qwen/Qwen3-8B --full
python -m experiment.paper.semantic.runner_openweight --model meta-llama/Llama-3.1-8B-Instruct --full

# Aggregate to LaTeX (writes to results/semantic/tables/)
python -m experiment.paper.semantic.make_table
```

### Metadata table

```bash
# OpenAI rows + the deterministic CiteExtract row (uses citeextract/verification/*)
python -m experiment.paper.metadata.runner --full

# Open-weight rows
python -m experiment.paper.metadata.runner_openweight --model Qwen/Qwen3-8B --full
python -m experiment.paper.metadata.runner_openweight --model meta-llama/Llama-3.1-8B-Instruct --full

# Aggregate (writes to results/metadata/tables/)
python -m experiment.paper.metadata.make_table
```

### Open-weight rows on a SLURM cluster

```bash
sbatch experiment/paper/_shared/run_openweight.sh Qwen/Qwen3-8B
sbatch experiment/paper/_shared/run_openweight.sh meta-llama/Llama-3.1-8B-Instruct
```

The script runs both tasks (semantic + metadata) for one model. Wall time ~1.5–3 hours per model on an L40S/A3090.

### Cross-task significance (bootstrap CIs + McNemar)

```bash
python -m experiment.paper.analysis.significance
# → results/significance/significance.json
# → results/significance/significance_summary.md
```

### Prevalence study (separate pipeline)

The prevalence study (CheckCitation pipeline run on 20 NeurIPS 2025 papers) lives entirely under [`prevalence/`](prevalence/) with its own `data/`, `results/`, and numbered scripts (`00_*` through `06_*`). See that folder for details.

## Notes

- **Resume:** every cell writes a partial JSONL to `results/{task}/partial/`. Re-running the same command picks up where it left off; pass `--no-resume` to start fresh.
- **Determinism:** OpenAI cells use `temperature=0` + JSON mode. Open-weight cells use greedy generation. Both are deterministic on re-run.
- **Production cell** (the bolded row in each table) is `gpt-4o-mini × title_abstract_passages` for semantic, and the deterministic CiteExtract cascade ([`metadata/run_production.py`](metadata/run_production.py)) for metadata.
- **Adding a new model:** edit *one* file per task — `runner.py` (API) or `runner_openweight.py` (HuggingFace) — there is no separate model-type folder.
