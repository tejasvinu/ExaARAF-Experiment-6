#!/bin/bash
# Resume training from checkpoint

python train.py \
    --dataset_name imdb \
    --model_name bert-base-uncased \
    --num_labels 2 \
    --output_dir ./results/imdb_bert_base_resumed \
    --resume_from_checkpoint ./results/imdb_bert_base/checkpoint-1000 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 32 \
    --learning_rate 2e-5 \
    --weight_decay 0.01 \
    --warmup_steps 500 \
    --logging_steps 100 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end true \
    --report_to tensorboard \
    --seed 42
