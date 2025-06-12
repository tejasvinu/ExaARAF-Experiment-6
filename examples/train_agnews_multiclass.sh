#!/bin/bash
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
    --save_strategy epoch \
    --load_best_model_at_end true \
    --metric_for_best_model eval_accuracy \
    --report_to tensorboard \
    --seed 42
