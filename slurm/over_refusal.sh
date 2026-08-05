#!/bin/bash
#SBATCH --job-name=over-refusal
#SBATCH --partition=gpu
#SBATCH --account=slurm-students
#SBATCH --output=slurm/logs/over_refusal_%j.out

# Submit with MODEL_IDX=0/1/2:
#   sbatch --export=MODEL_IDX=0 slurm/over_refusal.sh
#   sbatch --export=MODEL_IDX=1 slurm/over_refusal.sh
#   sbatch --export=MODEL_IDX=2 slurm/over_refusal.sh

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
ALPHAS=("20.0" "20.0" "5.0")

# Use same source vectors as defense experiment
JB_SRCS_LIST=(
    "en,de,zh,ja,ko,ru,th,ar,yo"
    "zh,ja,ko,ru,th,ar,yo,sw,am"
    "ko,th,ar,yo,sw,am"
)

# Test same target languages as defense
TARGET_LANGS="en,de,zh,ja,ko,ru,th,ar,yo,sw,am"

MODEL_IDX=${MODEL_IDX:-0}
MODEL_PATH=${MODEL_PATHS[$MODEL_IDX]}
MODEL_ALIAS=${MODEL_ALIASES[$MODEL_IDX]}
ALPHA=${ALPHAS[$MODEL_IDX]}
JB_SRCS=${JB_SRCS_LIST[$MODEL_IDX]}

echo "Model: $MODEL_ALIAS  alpha=$ALPHA  Start: $(date)"

cd ~/experiment_thesis
mkdir -p slurm/logs
source ~/thesis_experiment/Multilingual-Refusal/venv/bin/activate
export PYTHONPATH=/home/h24/baga0553/thesis_experiment/Multilingual-Refusal:/home/h24/baga0553/experiment_thesis:$PYTHONPATH

python scripts/over_refusal.py \
    --model_path     "$MODEL_PATH" \
    --model_alias    "$MODEL_ALIAS" \
    --vector_dir     "output/jailbreak_analysis/$MODEL_ALIAS" \
    --output_dir     "output/over_refusal/$MODEL_ALIAS" \
    --jb_srcs        "$JB_SRCS" \
    --target_langs   "$TARGET_LANGS" \
    --alpha          "$ALPHA" \
    --batch_size     8 \
    --max_new_tokens 200 \
    --max_test       100

echo "Done: $(date)"
