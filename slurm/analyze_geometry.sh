#!/bin/bash
#SBATCH --job-name=analyze-geometry
#SBATCH --partition=cpu
#SBATCH --account=slurm-students
#SBATCH --output=slurm/logs/geometry_%j.out

cd ~/experiment_thesis
echo "Start: $(date)"

source ~/thesis_experiment/Multilingual-Refusal/venv/bin/activate
export PYTHONPATH=/home/h24/baga0553/thesis_experiment/Multilingual-Refusal:/home/h24/baga0553/experiment_thesis:$PYTHONPATH

python scripts/analyze_geometry.py \
    --results_dir  output/jailbreak_analysis \
    --transfer_dir output/transfer \
    --defense_dir  output/defense \
    --output_dir   output/figures

echo "Done: $(date)"
