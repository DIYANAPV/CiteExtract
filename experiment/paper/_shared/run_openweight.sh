#!/bin/bash
# SLURM job script for the open-weight benchmark.
#
# Submits a one-GPU job that runs the full sweep (3 semantic + 1 metadata
# conditions) for ONE HuggingFace model. Pass the model id as $1.
#
# Examples:
#   sbatch experiment/paper/_shared/run_openweight.sh Qwen/Qwen3-8B
#   sbatch experiment/paper/_shared/run_openweight.sh meta-llama/Llama-3.1-8B-Instruct
#
# Output / errors are merged into:
#   experiment/paper/results/openweight_<jobid>_ow-bench.log
#
#SBATCH --job-name=ow-bench
#SBATCH --partition=p_48G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=experiment/paper/results/openweight_%j_%x.log
#SBATCH --error=experiment/paper/results/openweight_%j_%x.log

# NB: ``set -eu`` (no ``pipefail``). With ``pipefail`` an early
# ``nvidia-smi | head -10`` triggers SIGPIPE on nvidia-smi when head
# closes after 10 lines, which then aborts the whole script. We don't
# need pipe-failure detection here; Python failures still propagate
# normally because they're not in pipes.
set -eu

MODEL="${1:-Qwen/Qwen3-8B}"

echo "=== job $SLURM_JOB_ID: full open-weight benchmark for $MODEL ==="
echo "node: $(hostname)  start: $(date)"
nvidia-smi | head -10 || true

cd /nfs/home/muhammedd/Projects/phd/CHECKCITATION
source ~/vllm-env/bin/activate

# Make sure runtime deps are present (no-ops if already installed).
python -c "import accelerate" 2>/dev/null || pip install --quiet accelerate
python -c "import sentencepiece" 2>/dev/null || pip install --quiet sentencepiece

mkdir -p experiment/paper/results

# Semantic task: 3 conditions × 741 instances
python -m experiment.paper.semantic.runner_openweight --model "$MODEL" --full

# Metadata task: 1 condition × 302 instances
python -m experiment.paper.metadata.runner_openweight --model "$MODEL" --full

echo "=== done at $(date) ==="
