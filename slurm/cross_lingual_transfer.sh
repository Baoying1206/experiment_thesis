#!/bin/bash
#SBATCH --job-name=xling-transfer
#SBATCH --partition=gpu
#SBATCH --account=slurm-students
#SBATCH --output=slurm/logs/transfer_%j.out

# Submit with MODEL_IDX=0/1/2:
#   sbatch --export=MODEL_IDX=0 slurm/cross_lingual_transfer.sh

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
# Gemma uses alpha=5 to avoid catastrophic generation
ALPHAS=("20.0" "20.0" "5.0")

# RQ2: LRL sources (yo/sw/am) + HRL control sources (en/zh/ko)
# Script silently skips sources that have no jb_vec file
SOURCE_LANGS="en,zh,ko,yo,sw,am"

# Targets: high-refusal HRL + LRL (to test HRL j_l generalizing to LRL)
TARGET_LANGS="en,de,zh,ko,ja,yo,sw,am"

MODEL_IDX=${MODEL_IDX:-0}
MODEL_PATH=${MODEL_PATHS[$MODEL_IDX]}
MODEL_ALIAS=${MODEL_ALIASES[$MODEL_IDX]}
ALPHA=${ALPHAS[$MODEL_IDX]}

echo "Model: $MODEL_ALIAS  alpha=$ALPHA  Start: $(date)"

cd ~/thesis_experiment/Multilingual-Refusal
mkdir -p slurm/logs
source venv/bin/activate
export PYTHONPATH=/home/h24/baga0553/thesis_experiment/Multilingual-Refusal:$PYTHONPATH

python scripts/cross_lingual_transfer.py \
    --model_path      "$MODEL_PATH" \
    --model_alias     "$MODEL_ALIAS" \
    --vector_dir      "output/jailbreak_analysis/$MODEL_ALIAS" \
    --baseline_dir    "output/ja_vector_sweep" \
    --exp_id          "20250519-232436/1" \
    --output_dir      "output/transfer/$MODEL_ALIAS" \
    --source_langs    "$SOURCE_LANGS" \
    --target_langs    "$TARGET_LANGS" \
    --alpha           "$ALPHA" \
    --batch_size      8 \
    --max_new_tokens  200

echo "Done: $(date)"
