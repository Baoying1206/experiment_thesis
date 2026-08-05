#!/bin/bash
#SBATCH --job-name=rand-control
#SBATCH --partition=gpu
#SBATCH --account=slurm-students
#SBATCH --output=slurm/logs/random_control_%j.out

# Submit with MODEL_IDX=0/1/2 and MODE=attack/defense:
#   sbatch --export=MODEL_IDX=0,MODE=attack  slurm/random_vector_control.sh
#   sbatch --export=MODEL_IDX=0,MODE=defense slurm/random_vector_control.sh
#   sbatch --export=MODEL_IDX=1,MODE=attack  slurm/random_vector_control.sh
#   sbatch --export=MODEL_IDX=1,MODE=defense slurm/random_vector_control.sh
#   sbatch --export=MODEL_IDX=2,MODE=attack  slurm/random_vector_control.sh
#   sbatch --export=MODEL_IDX=2,MODE=defense slurm/random_vector_control.sh

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

# Compare dirs: where real j_src attack/defense results live
# For defense: output/defense/{alias}/defense_results.json
# For attack:  output/transfer/{alias}/{lang}/transfer_from_{src}.json
DEFENSE_COMPARE_DIRS=(
    "output/defense/Qwen2.5-7B-Instruct"
    "output/defense/Meta-Llama-3.1-8B-Instruct"
    "output/defense/gemma-2-9b-it"
)
ATTACK_COMPARE_DIRS=(
    "output/transfer/Qwen2.5-7B-Instruct"
    "output/transfer/Meta-Llama-3.1-8B-Instruct"
    "output/transfer/gemma-2-9b-it"
)

MODEL_IDX=${MODEL_IDX:-0}
MODE=${MODE:-attack}

MODEL_PATH=${MODEL_PATHS[$MODEL_IDX]}
MODEL_ALIAS=${MODEL_ALIASES[$MODEL_IDX]}
ALPHA=${ALPHAS[$MODEL_IDX]}

if [ "$MODE" = "defense" ]; then
    COMPARE_DIR=${DEFENSE_COMPARE_DIRS[$MODEL_IDX]}
else
    COMPARE_DIR=${ATTACK_COMPARE_DIRS[$MODEL_IDX]}
fi

TARGET_LANGS="en,de,zh,ja,ko,ru,th,ar,yo,sw,am"

echo "Model: $MODEL_ALIAS  mode=$MODE  alpha=$ALPHA  Start: $(date)"

cd ~/experiment_thesis
mkdir -p slurm/logs
source ~/thesis_experiment/Multilingual-Refusal/venv/bin/activate
export PYTHONPATH=/home/h24/baga0553/thesis_experiment/Multilingual-Refusal:/home/h24/baga0553/experiment_thesis:$PYTHONPATH

python scripts/random_vector_control.py \
    --mode          "$MODE" \
    --model_path    "$MODEL_PATH" \
    --model_alias   "$MODEL_ALIAS" \
    --vector_dir    "output/jailbreak_analysis/$MODEL_ALIAS" \
    --output_dir    "output/random_control/$MODEL_ALIAS" \
    --target_langs  "$TARGET_LANGS" \
    --alpha         "$ALPHA" \
    --n_seeds       5 \
    --batch_size    8 \
    --max_new_tokens 200 \
    --max_test      100 \
    --compare_dir   "$COMPARE_DIR"

echo "Done: $(date)"
