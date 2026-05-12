#!/bin/bash
#SBATCH --job-name=ow-bench
#SBATCH --partition=p_48G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=experiment/paper/results/openweight_%j_%x.log
#SBATCH --error=experiment/paper/results/openweight_%j_%x.log

set -eu

MODEL="${1:-Qwen/Qwen3-8B}"

if [[ -z "${REPO_ROOT:-}" ]]; then
    echo "error: REPO_ROOT is not set. Run: export REPO_ROOT=/path/to/checkcitation" >&2
    exit 2
fi

echo "=== job $SLURM_JOB_ID: full open-weight benchmark for $MODEL ==="
echo "node: $(hostname)  start: $(date)"
nvidia-smi | head -10 || true

cd "$REPO_ROOT"
source ~/vllm-env/bin/activate

python -c "import accelerate" 2>/dev/null || pip install --quiet accelerate
python -c "import sentencepiece" 2>/dev/null || pip install --quiet sentencepiece

mkdir -p experiment/paper/results

python -m experiment.paper.semantic.runner_openweight --model "$MODEL" --full

python -m experiment.paper.metadata.runner_openweight --model "$MODEL" --full

echo "=== done at $(date) ==="
