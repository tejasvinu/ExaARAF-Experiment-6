#!/bin/bash
#SBATCH --job-name=heavy_test_job
#SBATCH --output=heavy_test_job_output.log
#SBATCH --error=heavy_test_job_error.log
#SBATCH --time=03:00:00
#SBATCH --ntasks=24
# Multi-class text classification on AG News dataset

python train.py \
    --dataset_name ag_news \
    --text_column text \
    --label_column label \
    --model_name bert-base-uncased \
    --num_labels 4 \
    --output_dir ./results/agnews_bert_base \
    --num_train_epochs 3 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 64 \
    --learning_rate 3e-5 \
    --weight_decay 0.01 \
    --warmup_steps 1000 \
    --max_length 256 \
    --logging_steps 100 \
    --evaluation_strategy epoch \
    --save_strategy no \
    --load_best_model_at_end false \
    --metric_for_best_model eval_accuracy \
    --report_to tensorboard \
    --eval_on_train true \
    --seed 42
