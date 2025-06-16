#!/bin/bash
#SBATCH --job-name=heavy_test_job
#SBATCH --output=heavy_test_job_output.log
#SBATCH --error=heavy_test_job_error.log
#SBATCH --time=03:00:00
#SBATCH --ntasks=24
# Evaluate a fine-tuned model

python evaluate.py \
    --model_path ./results/imdb_bert_base \
    --dataset_name imdb \
    --test_split test \
    --batch_size 32 \
    --output_dir ./evaluation_results/imdb_bert_base \
    --save_predictions true
