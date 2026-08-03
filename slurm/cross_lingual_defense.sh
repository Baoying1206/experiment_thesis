#!/bin/bash
#SBATCH --job-name=cross-defense
#SBATCH --partition=gpu
#SBATCH --account=slurm-students
#SBATCH --output=slurm/logs/cross_defense_%A_%a.out
#SBATCH --array=0-8   # 3 models × 3 target langs

MODEL_PATHS=(
    "/home/h24/baga0553/models/Qwen2.5-7B-Instruct"
    "/home/h24/baga0553/models/Qwen2.5-7B-Instruct"
    "/home/h24/baga0553/models/Qwen2.5-7B-Instruct"
    "/home/h24/baga0553/models/Llama-3.1-8B-Instruct"
    "/home/h24/baga0553/models/Llama-3.1-8B-Instruct"
    "/home/h24/baga0553/models/Llama-3.1-8B-Instruct"
    "/home/h24/baga0553/models/gemma-2-9b-it"
    "/home/h24/baga0553/models/gemma-2-9b-it"
    "/home/h24/baga0553/models/gemma-2-9b-it"
)
MODEL_ALIASES=(
    "Qwen2.5-7B-Instruct"
    "Qwen2.5-7B-Instruct"
    "Qwen2.5-7B-Instruct"
    "Meta-Llama-3.1-8B-Instruct"
    "Meta-Llama-3.1-8B-Instruct"
    "Meta-Llama-3.1-8B-Instruct"
    "gemma-2-9b-it"
    "gemma-2-9b-it"
    "gemma-2-9b-it"
)
TARGET_LANGS=(
    "yo"  "th"  "ar"
    "yo"  "th"  "ar"
    "yo"  "th"  "ar"
)
ALPHAS=(
    "20.0" "20.0" "20.0"
    "20.0" "20.0" "20.0"
    "5.0"  "5.0"  "5.0"
)
# Qwen has refusal_dir_en.pt; LLaMA and Gemma only extracted for
# high-bypass languages — use ja (available for all three, bypass ~5-25%)
REFUSAL_SRCS=(
    "en"   "en"   "en"
    "ja"   "ja"   "ja"
    "ja"   "ja"   "ja"
)

MODEL_PATH=${MODEL_PATHS[$SLURM_ARRAY_TASK_ID]}
MODEL_ALIAS=${MODEL_ALIASES[$SLURM_ARRAY_TASK_ID]}
TARGET_LANG=${TARGET_LANGS[$SLURM_ARRAY_TASK_ID]}
ALPHA=${ALPHAS[$SLURM_ARRAY_TASK_ID]}
REFUSAL_SRC=${REFUSAL_SRCS[$SLURM_ARRAY_TASK_ID]}

echo "Model: $MODEL_ALIAS  Target: $TARGET_LANG  RefusalSrc: $REFUSAL_SRC  Alpha: $ALPHA  Start: $(date)"

cd ~/thesis_experiment/Multilingual-Refusal
source ~/thesis_experiment/Multilingual-Refusal/venv/bin/activate
export PYTHONPATH=/home/h24/baga0553/thesis_experiment/Multilingual-Refusal:$PYTHONPATH

python scripts/cross_lingual_defense.py \
    --model_path      "$MODEL_PATH" \
    --model_alias     "$MODEL_ALIAS" \
    --vector_dir      "output/jailbreak_analysis/$MODEL_ALIAS" \
    --baseline_dir    "output/ja_vector_sweep" \
    --exp_id          "20250519-232436/1" \
    --output_dir      "output/cross_defense/$MODEL_ALIAS" \
    --refusal_src     "$REFUSAL_SRC" \
    --jb_srcs         ja,ko,yo \
    --target_langs    "$TARGET_LANG" \
    --check_langs     en,de \
    --alpha           "$ALPHA" \
    --beta            1.0 \
    --batch_size      8 \
    --max_new_tokens  200 \
    --harmless_n      128

echo "Done: $(date)"
