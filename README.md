# BERT Fine-tuning Experiment

A comprehensive framework for fine-tuning BERT models on text classification tasks with extensive configuration options and detailed analysis tools.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration Options](#configuration-options)
- [Usage Examples](#usage-examples)
- [File Structure](#file-structure)
- [Detailed Documentation](#detailed-documentation)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## Overview

This project provides a flexible and comprehensive framework for fine-tuning BERT models on various text classification datasets. It includes extensive configuration options, evaluation tools, inference capabilities, and result analysis utilities.

### Key Components

- **Training Pipeline** (`train.py`): Complete training workflow with extensive configuration options
- **Evaluation Tools** (`evaluate.py`): Comprehensive model evaluation with detailed metrics and visualizations
- **Inference Engine** (`inference.py`): Easy-to-use inference on new texts
- **Result Analysis** (`analyze_results.py`): Detailed analysis of training results and metrics
- **Example Scripts**: Pre-configured examples for common use cases

## Features

### Training Features
- ✅ Support for any HuggingFace text classification dataset
- ✅ Extensive hyperparameter configuration via command-line arguments
- ✅ Multiple pre-trained model support (BERT, DistilBERT, RoBERTa, etc.)
- ✅ Advanced training features (early stopping, learning rate scheduling, mixed precision)
- ✅ Experiment tracking (Weights & Biases, TensorBoard)
- ✅ Checkpoint management and resuming
- ✅ Comprehensive logging and monitoring

### Evaluation Features
- ✅ Detailed metrics calculation (accuracy, precision, recall, F1)
- ✅ Confusion matrix and per-class analysis
- ✅ Visualization generation (confusion matrix, metrics plots, probability distributions)
- ✅ Misclassification analysis
- ✅ High-confidence prediction analysis

### Inference Features
- ✅ Single text and batch inference
- ✅ File-based input processing
- ✅ Probability scores and top-k predictions
- ✅ Multiple output formats (JSON, CSV)

### Analysis Features
- ✅ Training progress visualization
- ✅ Learning curve analysis
- ✅ Comprehensive experiment reporting
- ✅ Checkpoint analysis

## Installation

### Prerequisites
- Python 3.7 or higher
- CUDA-compatible GPU (optional but recommended)

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Optional: Install Weights & Biases for experiment tracking
```bash
pip install wandb
wandb login
```

## Quick Start

### 1. Basic Training on IMDB Dataset

```bash
python train.py \
    --dataset_name imdb \
    --model_name bert-base-uncased \
    --output_dir ./results/imdb_basic \
    --num_train_epochs 3 \
    --per_device_train_batch_size 16 \
    --learning_rate 2e-5
```

### 2. Evaluate the Trained Model

```bash
python evaluate.py \
    --model_path ./results/imdb_basic \
    --dataset_name imdb \
    --output_dir ./evaluation_results/imdb_basic
```

### 3. Run Inference on New Text

```bash
python inference.py \
    --model_path ./results/imdb_basic \
    --input_text "This movie was absolutely fantastic! Great acting and plot." \
    --return_probabilities true
```

### 4. Analyze Training Results

```bash
python analyze_results.py \
    --results_dir ./results/imdb_basic \
    --output_dir ./analysis_results/imdb_basic
```

## Configuration Options

The training script supports extensive configuration through command-line arguments:

### Dataset Configuration
- `--dataset_name`: HuggingFace dataset name (default: "imdb")
- `--dataset_config`: Dataset configuration name
- `--text_column`: Name of text column (default: "text")
- `--label_column`: Name of label column (default: "label")
- `--train_split`: Training data split name (default: "train")
- `--validation_split`: Validation data split name (default: "test")
- `--max_samples`: Maximum training samples to use
- `--validation_ratio`: Validation split ratio if no validation set (default: 0.1)

### Model Configuration
- `--model_name`: Pre-trained model name (default: "bert-base-uncased")
- `--num_labels`: Number of classification labels (default: 2)
- `--max_length`: Maximum sequence length (default: 512)
- `--truncation`: Whether to truncate sequences (default: True)
- `--padding`: Padding strategy (default: "max_length")

### Training Parameters
- `--num_train_epochs`: Number of training epochs (default: 3)
- `--per_device_train_batch_size`: Training batch size per device (default: 16)
- `--per_device_eval_batch_size`: Evaluation batch size per device (default: 16)
- `--learning_rate`: Learning rate (default: 2e-5)
- `--weight_decay`: Weight decay (default: 0.01)
- `--warmup_steps`: Number of warmup steps (default: 500)
- `--warmup_ratio`: Warmup ratio (default: 0.0)
- `--max_grad_norm`: Maximum gradient norm (default: 1.0)

### Advanced Training Options
- `--lr_scheduler_type`: Learning rate scheduler ("linear", "cosine", etc.)
- `--adam_beta1`: Adam beta1 parameter (default: 0.9)
- `--adam_beta2`: Adam beta2 parameter (default: 0.999)
- `--adam_epsilon`: Adam epsilon parameter (default: 1e-8)

### Evaluation and Saving
- `--evaluation_strategy`: Evaluation strategy ("no", "steps", "epoch")
- `--eval_steps`: Steps between evaluations (default: 500)
- `--save_strategy`: Save strategy ("no", "steps", "epoch")
- `--save_steps`: Steps between saves (default: 500)
- `--save_total_limit`: Maximum checkpoints to keep (default: 3)
- `--load_best_model_at_end`: Load best model at end (default: True)
- `--metric_for_best_model`: Metric for best model selection (default: "eval_accuracy")

### Early Stopping
- `--early_stopping_patience`: Early stopping patience (default: 3)
- `--early_stopping_threshold`: Early stopping threshold (default: 0.0)

### Experiment Tracking
- `--report_to`: Tracking service ("tensorboard", "wandb", "none")
- `--wandb_project`: Weights & Biases project name
- `--wandb_entity`: Weights & Biases entity name
- `--wandb_run_name`: Custom run name

### Hardware and Performance
- `--device`: Device to use ("auto", "cpu", "cuda")
- `--fp16`: Use mixed precision training (default: False)
- `--bf16`: Use bfloat16 precision (default: False)
- `--dataloader_num_workers`: Number of dataloader workers (default: 0)

### Reproducibility
- `--seed`: Random seed for reproducibility (default: 42)

## Usage Examples

### Example 1: Advanced Training with Weights & Biases

```bash
python train.py \
    --dataset_name imdb \
    --model_name bert-large-uncased \
    --output_dir ./results/imdb_bert_large \
    --num_train_epochs 5 \
    --per_device_train_batch_size 8 \
    --learning_rate 1e-5 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type cosine \
    --early_stopping_patience 3 \
    --report_to wandb \
    --wandb_project bert-finetuning \
    --fp16 true \
    --evaluation_strategy steps \
    --eval_steps 500 \
    --save_strategy steps \
    --save_steps 500
```

### Example 2: Multi-class Classification (AG News)

```bash
python train.py \
    --dataset_name ag_news \
    --model_name bert-base-uncased \
    --num_labels 4 \
    --output_dir ./results/agnews_bert \
    --max_length 256 \
    --per_device_train_batch_size 32 \
    --learning_rate 3e-5 \
    --num_train_epochs 3
```

### Example 3: Small Dataset Training

```bash
python train.py \
    --dataset_name imdb \
    --max_samples 1000 \
    --model_name distilbert-base-uncased \
    --output_dir ./results/imdb_small \
    --num_train_epochs 5 \
    --per_device_train_batch_size 16 \
    --learning_rate 5e-5 \
    --early_stopping_patience 2
```

### Example 4: Resume Training from Checkpoint

```bash
python train.py \
    --dataset_name imdb \
    --model_name bert-base-uncased \
    --output_dir ./results/imdb_resumed \
    --resume_from_checkpoint ./results/imdb_bert/checkpoint-1000 \
    --num_train_epochs 5
```

### Example 5: Comprehensive Evaluation

```bash
python evaluate.py \
    --model_path ./results/imdb_bert \
    --dataset_name imdb \
    --output_dir ./evaluation_results/imdb_bert \
    --batch_size 32 \
    --save_predictions true
```

### Example 6: Batch Inference from File

```bash
# Create a file with texts to classify
echo "This movie is amazing!" > input_texts.txt
echo "Terrible film, waste of time." >> input_texts.txt
echo "Average movie, nothing special." >> input_texts.txt

# Run inference
python inference.py \
    --model_path ./results/imdb_bert \
    --input_file input_texts.txt \
    --output_file predictions.json \
    --return_probabilities true \
    --batch_size 16
```

## File Structure

```
ExaARAF-Experiment-6/
├── train.py                 # Main training script
├── evaluate.py              # Model evaluation script
├── inference.py             # Inference script
├── analyze_results.py       # Results analysis script
├── create_examples.py       # Generate example scripts
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── examples/               # Example training scripts
│   ├── train_imdb_basic.sh
│   ├── train_imdb_advanced.sh
│   ├── train_agnews_multiclass.sh
│   ├── train_small_dataset.sh
│   ├── resume_training.sh
│   └── evaluate_model.sh
├── results/                # Training outputs (created during training)
│   └── [experiment_name]/
│       ├── config.json
│       ├── pytorch_model.bin
│       ├── tokenizer.json
│       ├── training_args.bin
│       ├── trainer_state.json
│       ├── experiment_config.json
│       ├── eval_results.json
│       └── checkpoint-*/
├── logs/                   # Training logs (created during training)
├── evaluation_results/     # Evaluation outputs (created during evaluation)
└── analysis_results/       # Analysis outputs (created during analysis)
```

## Detailed Documentation

### Training Process

The training script follows this workflow:

1. **Configuration**: Parse command-line arguments and validate settings
2. **Data Loading**: Load and preprocess the specified dataset
3. **Model Setup**: Initialize the pre-trained model and tokenizer
4. **Training Configuration**: Set up training arguments and callbacks
5. **Training**: Execute the training loop with monitoring
6. **Evaluation**: Evaluate on validation set and save results
7. **Model Saving**: Save the final model and tokenizer

### Evaluation Process

The evaluation script provides:

1. **Detailed Metrics**: Accuracy, precision, recall, F1-score (macro and weighted)
2. **Per-class Analysis**: Individual class performance metrics
3. **Confusion Matrix**: Visual representation of classification results
4. **Probability Analysis**: Distribution of prediction confidence scores
5. **Error Analysis**: Detailed examination of misclassified examples

### Supported Datasets

The framework works with any HuggingFace dataset that has text classification format. Popular examples:

- **Binary Classification**: IMDB, Stanford Sentiment Treebank (SST-2)
- **Multi-class Classification**: AG News, Yahoo Answers, DBpedia
- **Natural Language Inference**: SNLI, MultiNLI, RTE
- **Custom Datasets**: Any dataset with text and label columns

### Supported Models

Any HuggingFace model compatible with `AutoModelForSequenceClassification`:

- **BERT variants**: bert-base-uncased, bert-large-uncased, bert-base-cased
- **DistilBERT**: distilbert-base-uncased, distilbert-base-cased
- **RoBERTa**: roberta-base, roberta-large
- **ALBERT**: albert-base-v2, albert-large-v2
- **DeBERTa**: deberta-base, deberta-large
- **Custom Models**: Any fine-tuned model on HuggingFace Hub

## Advanced Features

### Experiment Tracking

#### Weights & Biases Integration
```bash
python train.py \
    --report_to wandb \
    --wandb_project my-bert-experiments \
    --wandb_entity my-team \
    --wandb_run_name imdb-bert-base-v1
```

#### TensorBoard Integration
```bash
python train.py --report_to tensorboard --logging_dir ./logs
tensorboard --logdir=./logs
```

### Mixed Precision Training

For faster training on modern GPUs:
```bash
python train.py --fp16 true  # For NVIDIA GPUs
python train.py --bf16 true  # For newer GPUs with bfloat16 support
```

### Learning Rate Scheduling

Available schedulers:
- `linear` (default): Linear decay with warmup
- `cosine`: Cosine annealing
- `cosine_with_restarts`: Cosine with warm restarts
- `polynomial`: Polynomial decay
- `constant`: Constant learning rate
- `constant_with_warmup`: Constant with warmup

### Early Stopping

Prevent overfitting with early stopping:
```bash
python train.py \
    --early_stopping_patience 3 \
    --early_stopping_threshold 0.001 \
    --metric_for_best_model eval_f1
```

## Troubleshooting

### Common Issues and Solutions

#### 1. CUDA Out of Memory
```bash
# Reduce batch size
--per_device_train_batch_size 8
--per_device_eval_batch_size 16

# Use gradient accumulation
--gradient_accumulation_steps 2

# Use mixed precision
--fp16 true
```

#### 2. Slow Training
```bash
# Increase batch size if memory allows
--per_device_train_batch_size 32

# Use multiple workers
--dataloader_num_workers 4

# Enable mixed precision
--fp16 true
```

#### 3. Poor Performance
```bash
# Adjust learning rate
--learning_rate 1e-5  # Lower for large models
--learning_rate 5e-5  # Higher for small datasets

# Increase training epochs
--num_train_epochs 5

# Add warmup
--warmup_ratio 0.1
```

#### 4. Dataset Loading Issues
```bash
# Specify dataset configuration
--dataset_config default

# Check column names
--text_column text
--label_column label

# Use custom splits
--train_split train
--validation_split validation
```

### Logging and Debugging

Enable debug logging:
```bash
python train.py --log_level DEBUG
```

Check training logs:
```bash
tail -f training.log
```

### Performance Optimization

#### For Large Datasets
```bash
python train.py \
    --max_samples 10000 \  # Limit training samples for testing
    --dataloader_num_workers 4 \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 1
```

#### For Small Datasets
```bash
python train.py \
    --num_train_epochs 10 \
    --learning_rate 5e-5 \
    --warmup_steps 100 \
    --early_stopping_patience 5
```

## Example Workflows

### Complete Experiment Workflow

```bash
# 1. Train the model
python train.py \
    --dataset_name imdb \
    --model_name bert-base-uncased \
    --output_dir ./results/imdb_experiment \
    --num_train_epochs 3 \
    --report_to wandb \
    --wandb_project bert-experiments

# 2. Evaluate the model
python evaluate.py \
    --model_path ./results/imdb_experiment \
    --dataset_name imdb \
    --output_dir ./evaluation_results/imdb_experiment

# 3. Run inference on new texts
python inference.py \
    --model_path ./results/imdb_experiment \
    --input_text "This movie was incredible!" \
    --return_probabilities true

# 4. Analyze the results
python analyze_results.py \
    --results_dir ./results/imdb_experiment \
    --output_dir ./analysis_results/imdb_experiment
```

### Hyperparameter Tuning Workflow

```bash
# Try different learning rates
for lr in 1e-5 2e-5 3e-5 5e-5; do
    python train.py \
        --dataset_name imdb \
        --model_name bert-base-uncased \
        --output_dir ./results/imdb_lr_${lr} \
        --learning_rate ${lr} \
        --run_name "imdb_lr_${lr}"
done

# Try different batch sizes
for bs in 8 16 32; do
    python train.py \
        --dataset_name imdb \
        --model_name bert-base-uncased \
        --output_dir ./results/imdb_bs_${bs} \
        --per_device_train_batch_size ${bs} \
        --run_name "imdb_bs_${bs}"
done
```

## Best Practices

### 1. Data Preparation
- Ensure your dataset has consistent text and label columns
- Check for class imbalance and consider stratified sampling
- Validate data quality (no missing values, consistent formatting)

### 2. Model Selection
- Start with `bert-base-uncased` for most English tasks
- Use `distilbert-base-uncased` for faster training/inference
- Consider `roberta-base` for better performance on some tasks

### 3. Hyperparameter Selection
- Learning rate: Start with 2e-5 for BERT, adjust based on dataset size
- Batch size: Use largest size that fits in memory
- Epochs: Start with 3-5, use early stopping to prevent overfitting

### 4. Training Strategy
- Use validation set for hyperparameter tuning
- Monitor training curves for overfitting
- Save multiple checkpoints for comparison

### 5. Evaluation
- Always evaluate on a held-out test set
- Look beyond accuracy: consider precision, recall, F1
- Analyze misclassified examples for insights

## Contributing

We welcome contributions! Please feel free to submit issues, feature requests, or pull requests.

### Development Setup
```bash
git clone <repository-url>
cd ExaARAF-Experiment-6
pip install -r requirements.txt
```

### Running Tests
```bash
# Test basic training
python train.py --max_samples 100 --num_train_epochs 1 --output_dir ./test_results

# Test evaluation
python evaluate.py --model_path ./test_results --output_dir ./test_evaluation

# Test inference
python inference.py --model_path ./test_results --input_text "Test text"
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Hugging Face Transformers library
- PyTorch team
- BERT paper authors
- Open source community

---

For more detailed information, please refer to the individual script documentation and example files in the `examples/` directory.
