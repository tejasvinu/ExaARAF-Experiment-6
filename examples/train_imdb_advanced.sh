#!/bin/bash
#SBATCH --job-name=heavy_test_job
#SBATCH --output=heavy_test_job_output.log
#SBATCH --error=heavy_test_job_error.log
#SBATCH --time=03:00:00
#SBATCH --ntasks=24

# Advanced training with Weights & Biases tracking
# Note: Tokenizer parallelism is automatically disabled to prevent fork warnings

python train.py \
    --dataset_name imdb \
    --model_name bert-large-uncased \
    --num_labels 2 \
    --output_dir ./results/imdb_bert_large_advanced \
    --num_train_epochs 5 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 16 \
    --learning_rate 1e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --adam_beta1 0.9 \
    --adam_beta2 0.999 \
    --adam_epsilon 1e-8 \
    --logging_steps 50 \
    --evaluation_strategy steps \
    --eval_steps 500 \
    --save_strategy no \
    --load_best_model_at_end false \
    --metric_for_best_model eval_f1 \
    --early_stopping_patience 3 \
    --early_stopping_threshold 0.001 \
    --report_to wandb \
    --wandb_project bert-finetuning-advanced \
    --run_name imdb_bert_large_cosine \
    --fp16 true \
    --dataloader_num_workers 2 \
    --eval_on_train true \
    --seed 42
