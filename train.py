import argparse
import os
import json
import logging
import random
import numpy as np
import torch
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import warnings # Added import

# Fix tokenizer parallelism warnings when using multiple dataloader workers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from datasets import Dataset, DatasetDict, load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments, 
    Trainer,
    EvalPrediction,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    set_seed
)
import wandb
from accelerate import Accelerator


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration."""
    warnings.filterwarnings('ignore') # Add this line to ignore warnings
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler()
        ]
    )
    logging.info("Tokenizer parallelism disabled to prevent fork warnings with dataloader workers")


def set_random_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="BERT Fine-tuning Experiment")
    
    # Dataset arguments
    parser.add_argument("--dataset_name", type=str, default="imdb", 
                       help="HuggingFace dataset name (default: imdb)")
    parser.add_argument("--dataset_config", type=str, default=None,
                       help="Dataset configuration name")
    parser.add_argument("--text_column", type=str, default="text",
                       help="Name of the text column in dataset")
    parser.add_argument("--label_column", type=str, default="label",
                       help="Name of the label column in dataset")
    parser.add_argument("--train_split", type=str, default="train",
                       help="Training data split name")
    parser.add_argument("--validation_split", type=str, default="test",
                       help="Validation data split name")
    parser.add_argument("--test_split", type=str, default=None,
                       help="Test data split name")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of samples to use for training")
    parser.add_argument("--validation_ratio", type=float, default=0.1,
                       help="Ratio of training data to use for validation if no validation split")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="bert-base-uncased",
                       help="Pre-trained model name or path")
    parser.add_argument("--num_labels", type=int, default=2,
                       help="Number of labels for classification")
    parser.add_argument("--max_length", type=int, default=512,
                       help="Maximum sequence length")
    parser.add_argument("--truncation", type=bool, default=True,
                       help="Whether to truncate sequences")
    parser.add_argument("--padding", type=str, default="max_length",
                       choices=["max_length", "longest", "do_not_pad"],
                       help="Padding strategy")
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, default="./results",
                       help="Output directory for model and logs")
    parser.add_argument("--num_train_epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=16,
                       help="Training batch size per device")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16,
                       help="Evaluation batch size per device")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                       help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="Weight decay")
    parser.add_argument("--warmup_steps", type=int, default=500,
                       help="Number of warmup steps")
    parser.add_argument("--warmup_ratio", type=float, default=0.0,
                       help="Warmup ratio")
    parser.add_argument("--adam_beta1", type=float, default=0.9,
                       help="Adam beta1 parameter")
    parser.add_argument("--adam_beta2", type=float, default=0.999,
                       help="Adam beta2 parameter")
    parser.add_argument("--adam_epsilon", type=float, default=1e-8,
                       help="Adam epsilon parameter")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                       help="Maximum gradient norm")
    
    # Scheduler arguments
    parser.add_argument("--lr_scheduler_type", type=str, default="linear",
                       choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
                       help="Learning rate scheduler type")
      # Evaluation arguments
    parser.add_argument("--evaluation_strategy", type=str, default="epoch",
                       choices=["no", "steps", "epoch"],
                       help="Evaluation strategy")
    parser.add_argument("--eval_steps", type=int, default=500,
                       help="Number of steps between evaluations")    # Checkpointing completely disabled to save space
    parser.add_argument("--save_strategy", type=str, default="no",
                       choices=["no"],
                       help="Save strategy (checkpointing disabled)")
    parser.add_argument("--load_best_model_at_end", type=bool, default=False,
                       help="Load best model at end (disabled to save space)")
    parser.add_argument("--metric_for_best_model", type=str, default="eval_accuracy",
                       help="Metric to use for best model selection")
    parser.add_argument("--greater_is_better", type=bool, default=True,
                       help="Whether higher metric values are better")
    
    # Early stopping arguments
    parser.add_argument("--early_stopping_patience", type=int, default=3,
                       help="Early stopping patience")
    parser.add_argument("--early_stopping_threshold", type=float, default=0.0,
                       help="Early stopping threshold")
    
    # Logging arguments
    parser.add_argument("--logging_dir", type=str, default="./logs",
                       help="Directory for logging")
    parser.add_argument("--logging_steps", type=int, default=100,
                       help="Number of steps between logs")
    parser.add_argument("--log_level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
      # Experiment tracking arguments
    parser.add_argument("--report_to", type=str, default="tensorboard",
                       choices=["tensorboard", "wandb", "none"],
                       help="Experiment tracking service")
    parser.add_argument("--wandb_project", type=str, default="bert-finetuning",
                       help="Weights & Biases project name")
    parser.add_argument("--wandb_entity", type=str, default=None,
                       help="Weights & Biases entity name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                       help="Weights & Biases run name")
    
    # Hardware arguments
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cpu", "cuda"],
                       help="Device to use for training")
    parser.add_argument("--fp16", type=bool, default=True,
                       help="Use mixed precision training (enabled by default for GPU performance)")
    parser.add_argument("--bf16", type=bool, default=False,
                       help="Use bfloat16 precision (use instead of fp16 on newer GPUs)")
    parser.add_argument("--dataloader_num_workers", type=int, default=2,
                       help="Number of dataloader workers (optimized to avoid tokenizer conflicts)")
    
    # Reproducibility arguments
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
      # Additional arguments
    parser.add_argument("--run_name", type=str, default=None,
                       help="Run name for experiment tracking")
    parser.add_argument("--do_train", type=bool, default=True,
                       help="Whether to run training")
    parser.add_argument("--do_eval", type=bool, default=True,
                       help="Whether to run evaluation")
    parser.add_argument("--do_predict", type=bool, default=False,
                       help="Whether to run prediction")
    parser.add_argument("--eval_on_train", type=bool, default=True,
                       help="Whether to evaluate on training set")
    
    return parser.parse_args()


def load_and_prepare_dataset(args: argparse.Namespace) -> Tuple[DatasetDict, List[str]]:
    """Load and prepare the dataset."""
    logging.info(f"Loading dataset: {args.dataset_name}")
    
    # Load dataset
    if args.dataset_config:
        dataset = load_dataset(args.dataset_name, args.dataset_config)
    else:
        dataset = load_dataset(args.dataset_name)
    
    # Get label names
    if hasattr(dataset[args.train_split].features[args.label_column], 'names'):
        label_names = dataset[args.train_split].features[args.label_column].names
    else:
        unique_labels = set(dataset[args.train_split][args.label_column])
        label_names = [f"label_{i}" for i in sorted(unique_labels)]
    
    logging.info(f"Label names: {label_names}")
    
    # Create train/validation split if needed
    if args.validation_split not in dataset:
        logging.info(f"Creating validation split with ratio {args.validation_ratio}")
        train_val = dataset[args.train_split].train_test_split(
            test_size=args.validation_ratio, 
            seed=args.seed,
            stratify_by_column=args.label_column
        )
        dataset[args.train_split] = train_val['train']
        dataset[args.validation_split] = train_val['test']
    
    # Limit samples if specified
    if args.max_samples:
        logging.info(f"Limiting training samples to {args.max_samples}")
        dataset[args.train_split] = dataset[args.train_split].select(range(args.max_samples))
    
    return dataset, label_names


def preprocess_function(examples: Dict[str, Any], tokenizer: AutoTokenizer, args: argparse.Namespace) -> Dict[str, Any]:
    """Preprocess function for tokenization."""
    # Tokenize texts without returning torch tensors to allow dynamic padding
    tokenized = tokenizer(
        examples[args.text_column],
        truncation=args.truncation,
        padding=args.padding,
        max_length=args.max_length
    )
    # Include labels if present
    if args.label_column in examples:
        tokenized['labels'] = examples[args.label_column]
    return tokenized


def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
    """Compute metrics for evaluation."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def create_trainer(
    model, 
    tokenizer, 
    train_dataset, 
    eval_dataset, 
    args: argparse.Namespace
) -> Trainer:
    """Create and configure the trainer."""
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        warmup_ratio=args.warmup_ratio,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_epsilon=args.adam_epsilon,
        max_grad_norm=args.max_grad_norm,        lr_scheduler_type=args.lr_scheduler_type,
        eval_strategy=args.evaluation_strategy,
        eval_steps=args.eval_steps,
        save_strategy="no",  # Completely disable checkpointing
        load_best_model_at_end=False,  # Disable to save space
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=args.greater_is_better,
        logging_dir=args.logging_dir,
        logging_steps=args.logging_steps,
        report_to=args.report_to,
        run_name=args.run_name or f"bert-finetuning-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        fp16=args.fp16,
        bf16=args.bf16,
        dataloader_num_workers=args.dataloader_num_workers,
        seed=args.seed,
        remove_unused_columns=False,
    )
    
    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Early stopping callback
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_threshold=args.early_stopping_threshold
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping_callback]
    )
    
    return trainer


def save_experiment_config(args: argparse.Namespace, output_dir: str) -> None:
    """Save experiment configuration to JSON file."""
    config_dict = vars(args)
    config_path = os.path.join(output_dir, "experiment_config.json")
    
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2, default=str)
    
    logging.info(f"Experiment configuration saved to {config_path}")


def setup_device(args: argparse.Namespace) -> str:
    """Setup and configure device for training."""
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
            gpu_count = torch.cuda.device_count()
            logging.info(f"CUDA is available! Using GPU(s). Device count: {gpu_count}")
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                gpu_capability = torch.cuda.get_device_properties(i).major
                logging.info(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB, Compute Capability: {gpu_capability}.x)")
                
                # Suggest optimal precision based on GPU architecture
                if gpu_capability >= 8:  # Ampere (A100, RTX 30xx) and newer
                    if not args.bf16:
                        logging.info(f"GPU {i} supports BF16 - consider using --bf16=True for better performance")
                elif gpu_capability >= 7:  # Volta (V100) and Turing (RTX 20xx)
                    if not args.fp16:
                        logging.info(f"GPU {i} supports FP16 - using --fp16=True for better performance")
        else:
            device = "cpu"
            logging.warning("CUDA is not available. Using CPU for training.")
            if args.fp16 or args.bf16:
                logging.warning("Mixed precision training disabled on CPU.")
                args.fp16 = False
                args.bf16 = False
    elif args.device == "cuda":
        if torch.cuda.is_available():
            device = "cuda"  
            gpu_count = torch.cuda.device_count()
            logging.info(f"Forced CUDA usage. Device count: {gpu_count}")
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                gpu_capability = torch.cuda.get_device_properties(i).major
                logging.info(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB, Compute Capability: {gpu_capability}.x)")
        else:
            logging.error("CUDA requested but not available! Falling back to CPU.")
            device = "cpu"
            args.fp16 = False
            args.bf16 = False
    else:
        device = "cpu"
        logging.info("Using CPU for training as requested.")
        if args.fp16 or args.bf16:
            logging.warning("Mixed precision training disabled on CPU.")
            args.fp16 = False
            args.bf16 = False
    
    return device


def log_gpu_memory_usage():
    """Log current GPU memory usage."""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            memory_allocated = torch.cuda.memory_allocated(i) / (1024**3)
            memory_reserved = torch.cuda.memory_reserved(i) / (1024**3)
            memory_total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            logging.info(f"GPU {i} Memory: {memory_allocated:.2f}GB allocated, "
                        f"{memory_reserved:.2f}GB reserved, {memory_total:.2f}GB total")


def main():
    """Main training function."""
    args = parse_arguments()
    
    # Setup
    setup_logging(args.log_level)
    set_random_seed(args.seed)
    
    # Setup device and optimize GPU settings
    device = setup_device(args)
    
    # Enable GPU optimizations if using CUDA
    if device == "cuda":
        # Enable TensorFloat-32 (TF32) on Ampere GPUs for better performance
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Enable cuDNN benchmark mode for consistent input sizes
        torch.backends.cudnn.benchmark = True
        
        # Optimize GPU memory usage
        torch.cuda.empty_cache()
        
        logging.info("GPU optimizations enabled: TF32, cuDNN benchmark")
    
    # Create unique output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_number = 1
    while True:
        output_dir_name = f"run_{run_number}_{timestamp}"
        current_output_dir = os.path.join(args.output_dir, output_dir_name)
        if not os.path.exists(current_output_dir):
            os.makedirs(current_output_dir)
            break
        run_number += 1
    
    args.output_dir = current_output_dir # Update args with the new output_dir
    args.logging_dir = os.path.join(current_output_dir, "logs") # Update logging_dir
    os.makedirs(args.logging_dir, exist_ok=True)

    logging.info(f"Output directory: {args.output_dir}") # Print the output directory
    print(f"Results will be stored in: {args.output_dir}")
      # Print GPU optimization summary
    if device == "cuda":
        print("🚀 GPU Training Configuration:")
        print(f"   ✓ Device: {device}")
        print(f"   ✓ Mixed Precision: FP16={args.fp16}, BF16={args.bf16}")
        print(f"   ✓ DataLoader Workers: {args.dataloader_num_workers} (tokenizer-safe)")
        print(f"   ✓ GPU Optimizations: TF32, cuDNN benchmark enabled")
        print(f"   ✓ Tokenizer Parallelism: Disabled (prevents fork warnings)")
    else:
        print(f"💻 CPU Training Configuration:")
        print(f"   ✓ Device: {device}")
        print(f"   ✓ DataLoader Workers: {args.dataloader_num_workers} (tokenizer-safe)")
        print(f"   ✓ Tokenizer Parallelism: Disabled")
    
    # Save experiment configuration
    save_experiment_config(args, args.output_dir)
    
    # Initialize wandb if specified
    if args.report_to == "wandb":
        wandb_run_name = args.wandb_run_name if args.wandb_run_name else output_dir_name
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=wandb_run_name,
            config=vars(args)
        )
    
    logging.info("Starting BERT fine-tuning experiment")
    logging.info(f"Arguments: {args}")
    
    # Load and prepare dataset
    dataset, label_names = load_and_prepare_dataset(args)
    
    # Load tokenizer and model
    logging.info(f"Loading model and tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=args.num_labels,
        id2label={i: label for i, label in enumerate(label_names)},
        label2id={label: i for i, label in enumerate(label_names)}
    )
    
    # Tokenize dataset
    logging.info("Tokenizing dataset")
    tokenized_dataset = dataset.map(
        lambda examples: preprocess_function(examples, tokenizer, args),
        batched=True,
        remove_columns=[args.text_column],  # Remove original text column
        desc="Tokenizing"
    )
    
    # Prepare datasets
    train_dataset = tokenized_dataset[args.train_split]
    eval_dataset = tokenized_dataset[args.validation_split]
    test_dataset = tokenized_dataset.get(args.test_split) if args.test_split else None
    
    # Create trainer
    trainer = create_trainer(model, tokenizer, train_dataset, eval_dataset, args)
      # Training
    if args.do_train:
        logging.info("Starting training")
        
        # Log initial GPU memory usage
        if device == "cuda":
            log_gpu_memory_usage()
        
        trainer.train()
        
        # Log final GPU memory usage
        if device == "cuda":
            log_gpu_memory_usage()
        
        # Save only final model (no checkpoints)
        trainer.save_model()
        tokenizer.save_pretrained(args.output_dir)
        
        logging.info(f"Training completed. Model saved to {args.output_dir}")
    
    # Evaluation on training set
    if args.do_train and args.eval_on_train:
        logging.info("Running evaluation on training set")
        train_eval_results = trainer.evaluate(eval_dataset=train_dataset)
        
        # Save training evaluation results
        train_eval_results_path = os.path.join(args.output_dir, "train_eval_results.json")
        with open(train_eval_results_path, 'w') as f:
            json.dump(train_eval_results, f, indent=2)
        
        logging.info(f"Training set evaluation results: {train_eval_results}")
        logging.info(f"Training set evaluation results saved to {train_eval_results_path}")
    
    # Evaluation on validation set
    if args.do_eval:
        logging.info("Running evaluation on validation set")
        eval_results = trainer.evaluate()
        
        # Save evaluation results
        eval_results_path = os.path.join(args.output_dir, "eval_results.json")
        with open(eval_results_path, 'w') as f:
            json.dump(eval_results, f, indent=2)
        
        logging.info(f"Validation set evaluation results: {eval_results}")
        logging.info(f"Validation set evaluation results saved to {eval_results_path}")
    
    # Prediction on test set
    if args.do_predict and test_dataset:
        logging.info("Running prediction on test set")
        predictions = trainer.predict(test_dataset)
        
        # Save predictions
        predictions_path = os.path.join(args.output_dir, "test_predictions.json")
        test_results = {
            'predictions': predictions.predictions.tolist(),
            'label_ids': predictions.label_ids.tolist(),
            'metrics': predictions.metrics
        }
        
        with open(predictions_path, 'w') as f:
            json.dump(test_results, f, indent=2)
        
        logging.info(f"Test predictions saved to {predictions_path}")
    
    # Finish wandb run
    if args.report_to == "wandb":
        wandb.finish()
    
    logging.info("Experiment completed successfully!")


if __name__ == "__main__":
    main()
