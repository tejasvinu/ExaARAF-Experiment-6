#!/bin/bash
#SBATCH --job-name=heavy_test_job
#SBATCH --output=heavy_test_job_output.log
#SBATCH --error=heavy_test_job_error.log
#SBATCH --time=03:00:00
#SBATCH --ntasks=24
# Training on a small subset for testing

python train.py \
    --dataset_name imdb \
    --max_samples 1000 \
    --model_name distilbert-base-uncased \
    --num_labels 2 \
    --output_dir ./results/imdb_distilbert_small \
    --num_train_epochs 5 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 32 \
    --learning_rate 5e-5 \
    --weight_decay 0.01 \
    --warmup_steps 100 \
    --logging_steps 50 \
    --evaluation_strategy epoch \
    --save_strategy no \
    --load_best_model_at_end false \
    --early_stopping_patience 2 \
    --report_to tensorboard \
    --eval_on_train true \
    --seed 42
