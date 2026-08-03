#!/bin/bash
#SBATCH --job-name=extract-jb
#SBATCH --partition=gpu
#SBATCH --account=slurm-students
#SBATCH --output=slurm/logs/extract_jb_%j.out

# Submit with MODEL_IDX=0/1/2:
#   sbatch --export=MODEL_IDX=0 slurm/extract_jb_vectors.sh

MODEL_PATHS=(
    "/home/h24/baga0553/models/Qwen2.5-7B-Instruct"
    "/home/h24/baga0553/models/Llama-3.1-8B-Instruct"
    "/home/h24/baga0553/models/gemma-2-9b-it"
)
MODEL_ALIASES=(
    "Qwen2.5-7B-Instruct"
    "Meta-Llama-3.1-8B-Instruct"
    "gemma-2-9b-it"
)

MODEL_IDX=${MODEL_IDX:-0}
MODEL_PATH=${MODEL_PATHS[$MODEL_IDX]}
MODEL_ALIAS=${MODEL_ALIASES[$MODEL_IDX]}

echo "Model: $MODEL_ALIAS  Start: $(date)"

cd ~/experiment_thesis
mkdir -p slurm/logs
source ~/thesis_experiment/Multilingual-Refusal/venv/bin/activate
export PYTHONPATH=/home/h24/baga0553/thesis_experiment/Multilingual-Refusal:/home/h24/baga0553/experiment_thesis:$PYTHONPATH

python scripts/extract_jailbreak_vectors.py \
    --model_path   "$MODEL_PATH" \
    --model_alias  "$MODEL_ALIAS" \
    --baseline_dir "output/ja_vector_sweep" \
    --exp_id       "20250519-232436/1" \
    --output_dir   "output/jailbreak_analysis/$MODEL_ALIAS" \
    --n_train      128 \
    --batch_size   8

echo "Done: $(date)"
