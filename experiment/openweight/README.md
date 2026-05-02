# Open-weight benchmark runner

Self-contained runner for adding open-weight model rows (Qwen 3, Llama 3.1, etc.) to the semantic and metadata tables.

**One process, one command** — uses HuggingFace `transformers` to load the model directly. No server, no daemon, no port forwarding. Just:

```bash
python -m experiment.openweight.runner --model Qwen/Qwen3-8B --task semantic --full
```

Designed to run on a SLURM/GPU compute node. Reads bundled benchmark JSONLs from `experiment/openweight/data/` (no `scp` needed). Writes per-cell CSVs into `experiment/results/`, matching the schema used by the laptop-side aggregators (`experiment/semantic_table/make_table.py` / `experiment/metadata_table/make_table.py`).

## What it covers

| Task | Conditions | Why |
|---|---|---|
| `semantic` | `title_only`, `title_abstract`, `title_abstract_passages` | Mirrors the OpenAI rows in `tab_semantic_results.tex`. |
| `metadata` | `llm_only` only | Open-weight models lack a native web-search tool, so the `llm_with_search` column stays an OpenAI-only column. |

## Cluster-side setup (one time, ~10 min)

### 1. Get an interactive GPU node

```bash
srun -p p_48G --gres=gpu:1 --time=08:00:00 --cpus-per-task=4 --mem=64G --pty bash
```

### 2. Activate a Python env with `torch` available

If you already have `~/vllm-env/` (which has torch + transformers), use it:

```bash
source ~/vllm-env/bin/activate
```

Otherwise create a fresh env:

```bash
uv venv --python 3.11 ~/openweight-env
source ~/openweight-env/bin/activate
uv pip install -r experiment/openweight/requirements.txt
```

### 3. Pull the latest code (and bundled data)

```bash
cd /nfs/home/muhammedd/Projects/phd/CiteExtract
git pull
ls experiment/openweight/   # expect: README.md, runner.py, data/, prompts/, requirements.txt
```

The benchmark JSONLs (`benchmark_enriched.jsonl`, `metadata_benchmark.jsonl`) ship with the repo under `experiment/openweight/data/` — total ~2.6 MB.

### 4. (Llama only) HuggingFace authentication

Llama 3.1 is a gated model — you need to accept its license once and provide a HuggingFace token. Skip this step if you only run Qwen 3.

```bash
# accept license at https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct first
huggingface-cli login   # or:  export HF_TOKEN=hf_xxxxxxxx
```

Qwen 3 is non-gated, no token needed.

## Running the benchmark

From the repo root on the cluster:

```bash
cd /nfs/home/muhammedd/Projects/phd/CiteExtract

# smoke test first (10 instances per cell — verifies everything wired up)
python -m experiment.openweight.runner --model Qwen/Qwen3-8B --task semantic --smoke
python -m experiment.openweight.runner --model Qwen/Qwen3-8B --task metadata --smoke

# full sweep (741 semantic × 3 conditions, then 302 metadata × 1 condition)
python -m experiment.openweight.runner --model Qwen/Qwen3-8B --task semantic --full
python -m experiment.openweight.runner --model Qwen/Qwen3-8B --task metadata --full

# repeat for Llama (requires HF login)
python -m experiment.openweight.runner --model meta-llama/Llama-3.1-8B-Instruct --task semantic --full
python -m experiment.openweight.runner --model meta-llama/Llama-3.1-8B-Instruct --task metadata --full
```

Each invocation:
1. Loads the model onto the GPU (~30 s for cached weights, ~5 min on first download from HuggingFace).
2. Iterates the requested instances, calling `model.generate()` for each.
3. Parses the JSON output, writes one row per instance to a CSV + a JSONL safety log.
4. Exits cleanly. Re-running the same command resumes from the safety log if it was interrupted.

Outputs:
- `experiment/results/semantic_Qwen3-8B_{condition}.csv` (3 files for semantic)
- `experiment/results/metadata_Qwen3-8B_llm_only.csv`
- `experiment/results/openweight_full_summary.json` (per-cell metrics)
- `experiment/results/*_partial_*.jsonl` (resume logs)

## Pulling results back to the laptop

After each full run completes, from your laptop:

```bash
scp 'muhammedd@cluster:/nfs/home/muhammedd/Projects/phd/CiteExtract/experiment/results/semantic_Qwen3-8B_*.csv' experiment/results/
scp 'muhammedd@cluster:/nfs/home/muhammedd/Projects/phd/CiteExtract/experiment/results/metadata_Qwen3-8B_*.csv' experiment/results/
scp 'muhammedd@cluster:/nfs/home/muhammedd/Projects/phd/CiteExtract/experiment/results/openweight_*_summary.json' experiment/results/
```

Then on your laptop, rerun `experiment/semantic_table/make_table.py` / `experiment/metadata_table/make_table.py` after adding the new model name to their `DEFAULT_MODELS` lists, and the LaTeX tables will pick up the new rows.

## Wall-time and VRAM expectations

On an RTX 3090 (24 GB) with Qwen 3 8B in fp16/bf16:

- VRAM used: ~16 GB weights + a few GB of KV cache.
- Throughput: ~0.5–1 generation/s for short prompts; passages cells slower (~0.3–0.5/s) due to 2k-token contexts.
- Full sweep (4,446 semantic + 302 metadata calls per model): expect **1.5–3 hours per model**.

## Notes

- **Qwen 3 thinking mode.** Qwen 3 emits `<think>...</think>` chain-of-thought by default, which would break our JSON-only output. The runner handles this three ways: (1) passes `enable_thinking=False` to the chat template, (2) prepends `/no_think` to user messages, (3) parser strips any `<think>...</think>` block that leaks past the first two.
- **Determinism.** Generation is greedy (`do_sample=False`, temperature=0) so re-running the same input gives the same output.
- **Concurrency.** Sequential — `transformers.generate()` is GPU-bound and one model instance can't usefully parallelize without batched generation. Wall time is dominated by per-call latency, not orchestration.
- **Resume.** A `*_partial_*.jsonl` is written per cell. Re-running the same command picks up where it left off; pass `--no-resume` to start fresh.
- **dtype.** Defaults to `auto` (transformers picks the model's native dtype, usually bf16 for recent Qwen / Llama). Force with `--dtype bf16` or `--dtype fp16` if needed.
