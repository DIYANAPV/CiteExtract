# Paper artifacts — CiteExtract benchmarks

all experiments , benchmarks conducted for paper lives in this folder.

## Layout

```
experiment/paper/
├── data/
├── prompts/
├── metadata/
├── semantic/
├── _shared/
├── analysis/
├── prevalence/
└── results/
    ├── metadata/{csv,partial,tables,summaries}/
    ├── semantic/{csv,partial,tables,summaries}/
    └── significance/
```

Each task folder is symmetric:

```
metadata/                       semantic/
├── runner.py                   ├── runner.py
├── runner_openweight.py        ├── runner_openweight.py
├── run_production.py           ├── llm_clients.py
├── llm_clients.py              ├── prompt_builder.py
├── prompt_builder.py           └── make_table.py
└── make_table.py
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


### Semantic table

```bash
# OpenAI rows (gpt-4o-mini, gpt-4o, gpt-5*, gpt-5.5*)
python -m experiment.paper.semantic.runner --full

# Open-weight rows 
python -m experiment.paper.semantic.runner_openweight --model Qwen/Qwen3-8B --full
python -m experiment.paper.semantic.runner_openweight --model meta-llama/Llama-3.1-8B-Instruct --full
```

### Metadata table

```bash
# OpenAI rows + the deterministic CiteExtract row (uses citeextract/verification/*)
python -m experiment.paper.metadata.runner --full

# Open-weight rows
python -m experiment.paper.metadata.runner_openweight --model Qwen/Qwen3-8B --full
python -m experiment.paper.metadata.runner_openweight --model meta-llama/Llama-3.1-8B-Instruct --full
```

### Open-weight rows on a SLURM cluster

```bash
sbatch experiment/paper/_shared/run_openweight.sh Qwen/Qwen3-8B
sbatch experiment/paper/_shared/run_openweight.sh meta-llama/Llama-3.1-8B-Instruct
```

