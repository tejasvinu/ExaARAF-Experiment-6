#!/bin/bash
# Evaluate a fine-tuned model

python evaluate.py \
    --model_path ./results/imdb_bert_base \
    --dataset_name imdb \
    --test_split test \
    --batch_size 32 \
    --output_dir ./evaluation_results/imdb_bert_base \
    --save_predictions true
