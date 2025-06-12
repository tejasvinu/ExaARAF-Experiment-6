#!/usr/bin/env python3
"""
Example training scripts for different scenarios
"""
import os

def create_example_scripts():
    """Create example training scripts for different use cases."""
    
    # Basic IMDB sentiment analysis
    imdb_script = """#!/bin/bash
# Basic IMDB sentiment analysis training

python train.py \\
    --dataset_name imdb \\
    --model_name bert-base-uncased \\
    --num_labels 2 \\
    --output_dir ./results/imdb_bert_base \\
    --num_train_epochs 3 \\
    --per_device_train_batch_size 16 \\
    --per_device_eval_batch_size 32 \\
    --learning_rate 2e-5 \\
    --weight_decay 0.01 \\
    --warmup_steps 500 \\
    --logging_steps 100 \\
    --evaluation_strategy epoch \\
    --save_strategy epoch \\
    --load_best_model_at_end true \\
    --report_to tensorboard \\
    --seed 42
"""
    
    # Advanced configuration with wandb
    advanced_script = """#!/bin/bash
# Advanced training with Weights & Biases tracking

python train.py \\
    --dataset_name imdb \\
    --model_name bert-large-uncased \\
    --num_labels 2 \\
    --output_dir ./results/imdb_bert_large_advanced \\
    --num_train_epochs 5 \\
    --per_device_train_batch_size 8 \\
    --per_device_eval_batch_size 16 \\
    --learning_rate 1e-5 \\
    --weight_decay 0.01 \\
    --warmup_ratio 0.1 \\
    --lr_scheduler_type cosine \\
    --max_grad_norm 1.0 \\
    --adam_beta1 0.9 \\
    --adam_beta2 0.999 \\
    --adam_epsilon 1e-8 \\
    --logging_steps 50 \\
    --evaluation_strategy steps \\
    --eval_steps 500 \\
    --save_strategy steps \\
    --save_steps 500 \\
    --save_total_limit 3 \\
    --load_best_model_at_end true \\
    --metric_for_best_model eval_f1 \\
    --early_stopping_patience 3 \\
    --early_stopping_threshold 0.001 \\
    --report_to wandb \\
    --wandb_project bert-finetuning-advanced \\
    --run_name imdb_bert_large_cosine \\
    --fp16 true \\
    --seed 42
"""
    
    # Multi-class classification (AG News)
    multiclass_script = """#!/bin/bash
# Multi-class text classification on AG News dataset

python train.py \\
    --dataset_name ag_news \\
    --text_column text \\
    --label_column label \\
    --model_name bert-base-uncased \\
    --num_labels 4 \\
    --output_dir ./results/agnews_bert_base \\
    --num_train_epochs 3 \\
    --per_device_train_batch_size 32 \\
    --per_device_eval_batch_size 64 \\
    --learning_rate 3e-5 \\
    --weight_decay 0.01 \\
    --warmup_steps 1000 \\
    --max_length 256 \\
    --logging_steps 100 \\
    --evaluation_strategy epoch \\
    --save_strategy epoch \\
    --load_best_model_at_end true \\
    --metric_for_best_model eval_accuracy \\
    --report_to tensorboard \\
    --seed 42
"""
    
    # Small dataset with limited samples
    small_dataset_script = """#!/bin/bash
# Training on a small subset for testing

python train.py \\
    --dataset_name imdb \\
    --max_samples 1000 \\
    --model_name distilbert-base-uncased \\
    --num_labels 2 \\
    --output_dir ./results/imdb_distilbert_small \\
    --num_train_epochs 5 \\
    --per_device_train_batch_size 16 \\
    --per_device_eval_batch_size 32 \\
    --learning_rate 5e-5 \\
    --weight_decay 0.01 \\
    --warmup_steps 100 \\
    --logging_steps 50 \\
    --evaluation_strategy epoch \\
    --save_strategy epoch \\
    --load_best_model_at_end true \\
    --early_stopping_patience 2 \\
    --report_to tensorboard \\
    --seed 42
"""
    
    # Resume from checkpoint
    resume_script = """#!/bin/bash
# Resume training from checkpoint

python train.py \\
    --dataset_name imdb \\
    --model_name bert-base-uncased \\
    --num_labels 2 \\
    --output_dir ./results/imdb_bert_base_resumed \\
    --resume_from_checkpoint ./results/imdb_bert_base/checkpoint-1000 \\
    --num_train_epochs 5 \\
    --per_device_train_batch_size 16 \\
    --per_device_eval_batch_size 32 \\
    --learning_rate 2e-5 \\
    --weight_decay 0.01 \\
    --warmup_steps 500 \\
    --logging_steps 100 \\
    --evaluation_strategy epoch \\
    --save_strategy epoch \\
    --load_best_model_at_end true \\
    --report_to tensorboard \\
    --seed 42
"""
    
    # Evaluation script
    eval_script = """#!/bin/bash
# Evaluate a fine-tuned model

python evaluate.py \\
    --model_path ./results/imdb_bert_base \\
    --dataset_name imdb \\
    --test_split test \\
    --batch_size 32 \\
    --output_dir ./evaluation_results/imdb_bert_base \\
    --save_predictions true
"""
    
    return {
        'train_imdb_basic.sh': imdb_script,
        'train_imdb_advanced.sh': advanced_script,
        'train_agnews_multiclass.sh': multiclass_script,
        'train_small_dataset.sh': small_dataset_script,
        'resume_training.sh': resume_script,
        'evaluate_model.sh': eval_script
    }

if __name__ == "__main__":
    scripts = create_example_scripts()
    
    # Create examples directory
    os.makedirs("examples", exist_ok=True)
    
    # Write scripts
    for filename, content in scripts.items():
        filepath = os.path.join("examples", filename)
        with open(filepath, 'w') as f:
            f.write(content.strip() + '\n')
        
        # Make executable on Unix-like systems
        try:
            os.chmod(filepath, 0o755)
        except:
            pass  # Windows doesn't support chmod
    
    print("Example scripts created in 'examples' directory:")
    for filename in scripts.keys():
        print(f"  - {filename}")
