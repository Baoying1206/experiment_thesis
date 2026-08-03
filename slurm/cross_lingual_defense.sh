#!/bin/bash
#SBATCH --job-name=xling-defense
#SBATCH --partition=gpu
#SBATCH --account=slurm-students
#SBATCH --output=slurm/logs/defense_%j.out

# Submit with MODEL_IDX=0/1/2:
#   sbatch --export=MODEL_IDX=0 slurm/cross_lingual_defense.sh

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

# RQ3: subtract j_l to restore refusal on bypassed samples.
# HRL sources (en/zh/ja/ko) are reliable; LRL (yo/sw/am) included but may
# have no jb_vec if bypass ≈ 100% — script skips missing ones automatically.
# Diagonal entries replicate base-paper Table 6; off-diagonal = new results.
JB_SRCS="en,de,zh,ja,ko,ru,th,ar,yo,sw,am"
TARGET_LANGS="en,de,zh,ja,ko,ru,th,ar,es,fr,it,nl,pl,yo,sw,am"

MODEL_IDX=${MODEL_IDX:-0}
MODEL_PATH=${MODEL_PATHS[$MODEL_IDX]}
MODEL_ALIAS=${MODEL_ALIASES[$MODEL_IDX]}
ALPHA=${ALPHAS[$MODEL_IDX]}

echo "Model: $MODEL_ALIAS  alpha=$ALPHA  Start: $(date)"

cd ~/experiment_thesis
mkdir -p slurm/logs
source ~/thesis_experiment/Multilingual-Refusal/venv/bin/activate
export PYTHONPATH=/home/h24/baga0553/thesis_experiment/Multilingual-Refusal:/home/h24/baga0553/experiment_thesis:$PYTHONPATH

python scripts/cross_lingual_defense.py \
    --model_path      "$MODEL_PATH" \
    --model_alias     "$MODEL_ALIAS" \
    --vector_dir      "output/jailbreak_analysis/$MODEL_ALIAS" \
    --baseline_dir    "output/ja_vector_sweep" \
    --exp_id          "20250519-232436/1" \
    --output_dir      "output/defense/$MODEL_ALIAS" \
    --jb_srcs         "$JB_SRCS" \
    --target_langs    "$TARGET_LANGS" \
    --alpha           "$ALPHA" \
    --batch_size      8 \
    --max_new_tokens  200

echo "Done: $(date)"
