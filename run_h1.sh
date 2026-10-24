#!/bin/bash
# H1 relation extraction on Kaya (UWA HPC): one SLURM array task per model.
#
#   array index -> model (see MODELS in src/run_re.py)
#     0 Llama-3.1-8B-Instruct   1 Qwen2.5-7B-Instruct
#     2 Qwen2.5-14B-Instruct    3 Ophtimus-Instruct-8B
#
# All four run on the same GPU type, otherwise the timing table is not
# comparable. Submit from the repo root:
#
#   sbatch run_h1.sh                       # zero-shot
#   PROMPT=grounded sbatch run_h1.sh       # few-shot
#
#SBATCH --account=pmc091
#SBATCH --partition=ondemand-gpu
#SBATCH --gres=gpu:v100-32gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --job-name=h1_re
#SBATCH --array=0-3
#SBATCH --output=logs/%x_%A_%a.log

set -euo pipefail
source /uwahpc/rocky9/python/anaconda3/2024.06/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-base}"

# Compute nodes have no internet: models must already be in the HF cache.
export HF_HOME="${HF_HOME:-/group/pmc091/$USER/hf_cache}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs

echo "Task $SLURM_ARRAY_TASK_ID on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
# fail fast if torch cannot see the GPU (silent CPU fallback is the usual bug)
python -c "import torch,sys; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); sys.exit(0 if torch.cuda.is_available() else 'FATAL: no CUDA')"

python src/run_re.py \
  --backend hf \
  --model-index "$SLURM_ARRAY_TASK_ID" \
  --prompt "${PROMPT:-zero}" \
  --input "${INPUT:-data/preprocessed.jsonl}" \
  --outdir "${OUTDIR:-results/h1}" \
  --max-new-tokens 1024
