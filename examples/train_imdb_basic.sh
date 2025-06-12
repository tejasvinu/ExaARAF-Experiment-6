#!/bin/bash
#SBATCH --job-name=heavy_test_job
#SBATCH --output=heavy_test_job_output.log
#SBATCH --error=heavy_test_job_error.log
#SBATCH --time=03:00:00
#SBATCH --ntasks=24

# Basic IMDB sentiment analysis training

/opt/miniforge3/bin/python train.py \
    --dataset_name imdb \
    --model_name bert-base-uncased \
    --num_labels 2 \
    --output_dir ./results/imdb_bert_base \
    --num_train_epochs 3 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 32 \
    --learning_rate 2e-5 \
    --weight_decay 0.01 \
    --warmup_steps 500 \
    --logging_steps 100 \
    --evaluation_strategy epoch \
    --save_strategy no \
    --load_best_model_at_end true \
    --report_to tensorboard \
    --seed 42
