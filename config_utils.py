# Configuration Files and Additional Scripts

## Configuration Templates

### config.yaml - Training Configuration Template
```yaml
# Dataset Configuration
dataset:
  name: "imdb"
  config: null
  text_column: "text"
  label_column: "label"
  train_split: "train"
  validation_split: "test"
  test_split: null
  max_samples: null
  validation_ratio: 0.1

# Model Configuration
model:
  name: "bert-base-uncased"
  num_labels: 2
  max_length: 512
  truncation: true
  padding: "max_length"

# Training Configuration
training:
  output_dir: "./results"
  num_train_epochs: 3
  per_device_train_batch_size: 16
  per_device_eval_batch_size: 16
  learning_rate: 2e-5
  weight_decay: 0.01
  warmup_steps: 500
  warmup_ratio: 0.0
  max_grad_norm: 1.0
  lr_scheduler_type: "linear"

# Evaluation Configuration
evaluation:
  evaluation_strategy: "epoch"
  eval_steps: 500
  save_strategy: "epoch"
  save_steps: 500
  save_total_limit: 3
  load_best_model_at_end: true
  metric_for_best_model: "eval_accuracy"
  greater_is_better: true

# Early Stopping
early_stopping:
  patience: 3
  threshold: 0.0

# Logging and Tracking
logging:
  logging_dir: "./logs"
  logging_steps: 100
  log_level: "INFO"
  report_to: "tensorboard"

# Hardware and Performance
hardware:
  device: "auto"
  fp16: false
  bf16: false
  dataloader_num_workers: 0

# Reproducibility
reproducibility:
  seed: 42

# Experiment Tracking (Weights & Biases)
wandb:
  project: "bert-finetuning"
  entity: null
  run_name: null
```

## Utility Scripts

### utils.py - Common Utilities
```python
import os
import json
import yaml
import logging
import random
import numpy as np
import torch
from typing import Dict, Any, Optional, List
from pathlib import Path


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            return yaml.safe_load(f)
        else:
            return json.load(f)


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """Save configuration to file."""
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    with open(config_path, 'w') as f:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            yaml.dump(config, f, default_flow_style=False, indent=2)
        else:
            json.dump(config, f, indent=2, default=str)


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> None:
    """Setup comprehensive logging configuration."""
    handlers = [logging.StreamHandler()]
    
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def set_random_seeds(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_arg: str = "auto") -> torch.device:
    """Get the appropriate device for training/inference."""
    if device_arg == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logging.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
        else:
            device = torch.device("cpu")
            logging.info("Using CPU device")
    elif device_arg == "cuda":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logging.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
        else:
            logging.warning("CUDA requested but not available, using CPU")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
        logging.info("Using CPU device")
    
    return device


def count_parameters(model) -> Dict[str, int]:
    """Count the number of parameters in a model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params
    }


def get_model_size(model) -> str:
    """Get human-readable model size."""
    param_count = count_parameters(model)['total_parameters']
    
    if param_count >= 1e9:
        return f"{param_count / 1e9:.1f}B"
    elif param_count >= 1e6:
        return f"{param_count / 1e6:.1f}M"
    elif param_count >= 1e3:
        return f"{param_count / 1e3:.1f}K"
    else:
        return str(param_count)


def create_experiment_name(config: Dict[str, Any]) -> str:
    """Create a descriptive experiment name from configuration."""
    dataset_name = config.get('dataset', {}).get('name', 'unknown')
    model_name = config.get('model', {}).get('name', 'unknown').split('/')[-1]
    lr = config.get('training', {}).get('learning_rate', 'unknown')
    batch_size = config.get('training', {}).get('per_device_train_batch_size', 'unknown')
    epochs = config.get('training', {}).get('num_train_epochs', 'unknown')
    
    return f"{dataset_name}_{model_name}_lr{lr}_bs{batch_size}_epochs{epochs}"


def ensure_dir(directory: str) -> None:
    """Ensure directory exists."""
    os.makedirs(directory, exist_ok=True)


def get_git_commit_hash() -> Optional[str]:
    """Get current git commit hash if available."""
    try:
        import subprocess
        result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
    except:
        pass
    return None


def log_system_info() -> None:
    """Log system information for reproducibility."""
    import platform
    import sys
    
    logging.info("System Information:")
    logging.info(f"  Platform: {platform.platform()}")
    logging.info(f"  Python: {sys.version}")
    logging.info(f"  PyTorch: {torch.__version__}")
    
    if torch.cuda.is_available():
        logging.info(f"  CUDA: {torch.version.cuda}")
        logging.info(f"  GPU: {torch.cuda.get_device_name()}")
        logging.info(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    git_hash = get_git_commit_hash()
    if git_hash:
        logging.info(f"  Git Commit: {git_hash}")


class ExperimentTracker:
    """Simple experiment tracking utility."""
    
    def __init__(self, experiment_dir: str):
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.metrics = {}
        self.config = {}
    
    def log_config(self, config: Dict[str, Any]) -> None:
        """Log experiment configuration."""
        self.config = config
        config_path = self.experiment_dir / "config.json"
        save_config(config, str(config_path))
    
    def log_metric(self, name: str, value: float, step: int = None) -> None:
        """Log a metric value."""
        if name not in self.metrics:
            self.metrics[name] = []
        
        metric_entry = {'value': value}
        if step is not None:
            metric_entry['step'] = step
        
        self.metrics[name].append(metric_entry)
    
    def log_metrics(self, metrics: Dict[str, float], step: int = None) -> None:
        """Log multiple metrics."""
        for name, value in metrics.items():
            self.log_metric(name, value, step)
    
    def save_metrics(self) -> None:
        """Save all logged metrics."""
        metrics_path = self.experiment_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2, default=str)
    
    def log_artifact(self, file_path: str, artifact_name: str = None) -> None:
        """Copy an artifact to the experiment directory."""
        import shutil
        
        source_path = Path(file_path)
        if artifact_name is None:
            artifact_name = source_path.name
        
        dest_path = self.experiment_dir / "artifacts" / artifact_name
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        if source_path.is_file():
            shutil.copy2(source_path, dest_path)
        elif source_path.is_dir():
            shutil.copytree(source_path, dest_path, dirs_exist_ok=True)


def format_time(seconds: float) -> str:
    """Format time in human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage."""
    import psutil
    
    # System memory
    memory = psutil.virtual_memory()
    
    result = {
        'system_memory_total_gb': memory.total / 1e9,
        'system_memory_used_gb': memory.used / 1e9,
        'system_memory_percent': memory.percent
    }
    
    # GPU memory
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        gpu_allocated = torch.cuda.memory_allocated()
        gpu_reserved = torch.cuda.memory_reserved()
        
        result.update({
            'gpu_memory_total_gb': gpu_memory / 1e9,
            'gpu_memory_allocated_gb': gpu_allocated / 1e9,
            'gpu_memory_reserved_gb': gpu_reserved / 1e9,
            'gpu_memory_allocated_percent': (gpu_allocated / gpu_memory) * 100
        })
    
    return result


def validate_config(config: Dict[str, Any]) -> List[str]:
    """Validate configuration and return list of warnings/errors."""
    warnings = []
    
    # Check required sections
    required_sections = ['dataset', 'model', 'training']
    for section in required_sections:
        if section not in config:
            warnings.append(f"Missing required section: {section}")
    
    # Validate dataset config
    if 'dataset' in config:
        dataset_config = config['dataset']
        if not dataset_config.get('name'):
            warnings.append("Dataset name is required")
    
    # Validate model config
    if 'model' in config:
        model_config = config['model']
        if not model_config.get('name'):
            warnings.append("Model name is required")
        
        num_labels = model_config.get('num_labels', 2)
        if num_labels < 2:
            warnings.append("num_labels should be at least 2")
    
    # Validate training config
    if 'training' in config:
        training_config = config['training']
        
        lr = training_config.get('learning_rate', 2e-5)
        if lr <= 0 or lr > 1:
            warnings.append(f"Learning rate {lr} seems unusual")
        
        epochs = training_config.get('num_train_epochs', 3)
        if epochs <= 0:
            warnings.append("Number of epochs should be positive")
        
        batch_size = training_config.get('per_device_train_batch_size', 16)
        if batch_size <= 0:
            warnings.append("Batch size should be positive")
    
    return warnings
```

### data_utils.py - Data Processing Utilities
```python
import os
import json
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datasets import Dataset, DatasetDict, load_dataset
import logging


def load_custom_dataset(data_path: str, text_column: str = 'text', 
                       label_column: str = 'label') -> DatasetDict:
    """Load a custom dataset from various file formats."""
    
    if data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
    elif data_path.endswith('.json') or data_path.endswith('.jsonl'):
        df = pd.read_json(data_path, lines=data_path.endswith('.jsonl'))
    elif data_path.endswith('.tsv'):
        df = pd.read_csv(data_path, sep='\t')
    else:
        raise ValueError(f"Unsupported file format: {data_path}")
    
    # Validate columns
    if text_column not in df.columns:
        raise ValueError(f"Text column '{text_column}' not found in dataset")
    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found in dataset")
    
    # Convert to Dataset
    dataset = Dataset.from_pandas(df)
    
    # Create train/test split
    dataset = dataset.train_test_split(test_size=0.2, seed=42, 
                                     stratify_by_column=label_column)
    
    return DatasetDict({
        'train': dataset['train'],
        'test': dataset['test']
    })


def analyze_dataset(dataset: DatasetDict, text_column: str = 'text', 
                   label_column: str = 'label') -> Dict[str, Any]:
    """Analyze dataset statistics."""
    
    analysis = {}
    
    for split_name, split_data in dataset.items():
        split_analysis = {
            'num_samples': len(split_data),
            'text_stats': {},
            'label_stats': {}
        }
        
        # Text statistics
        texts = split_data[text_column]
        text_lengths = [len(text.split()) for text in texts]
        
        split_analysis['text_stats'] = {
            'min_length': min(text_lengths),
            'max_length': max(text_lengths),
            'avg_length': sum(text_lengths) / len(text_lengths),
            'median_length': sorted(text_lengths)[len(text_lengths) // 2]
        }
        
        # Label statistics
        labels = split_data[label_column]
        label_counts = {}
        for label in labels:
            label_counts[label] = label_counts.get(label, 0) + 1
        
        split_analysis['label_stats'] = {
            'label_counts': label_counts,
            'num_classes': len(label_counts),
            'class_balance': {k: v/len(labels) for k, v in label_counts.items()}
        }
        
        analysis[split_name] = split_analysis
    
    return analysis


def create_data_subset(dataset: DatasetDict, max_samples_per_split: int, 
                      label_column: str = 'label', seed: int = 42) -> DatasetDict:
    """Create a subset of the dataset for quick testing."""
    
    subset_dataset = DatasetDict()
    
    for split_name, split_data in dataset.items():
        if len(split_data) > max_samples_per_split:
            # Stratified sampling to maintain class balance
            subset_data = split_data.train_test_split(
                train_size=max_samples_per_split,
                seed=seed,
                stratify_by_column=label_column
            )['train']
        else:
            subset_data = split_data
        
        subset_dataset[split_name] = subset_data
    
    return subset_dataset


def balance_dataset(dataset: Dataset, label_column: str = 'label', 
                   method: str = 'oversample') -> Dataset:
    """Balance dataset classes."""
    
    # Get label counts
    labels = dataset[label_column]
    label_counts = {}
    for label in labels:
        label_counts[label] = label_counts.get(label, 0) + 1
    
    if method == 'oversample':
        # Oversample minority classes to match majority class
        max_count = max(label_counts.values())
        
        balanced_indices = []
        for label in label_counts:
            label_indices = [i for i, l in enumerate(labels) if l == label]
            
            # Repeat indices to reach max_count
            multiplier = max_count // len(label_indices)
            remainder = max_count % len(label_indices)
            
            balanced_indices.extend(label_indices * multiplier)
            balanced_indices.extend(label_indices[:remainder])
        
        return dataset.select(balanced_indices)
    
    elif method == 'undersample':
        # Undersample majority classes to match minority class
        min_count = min(label_counts.values())
        
        balanced_indices = []
        for label in label_counts:
            label_indices = [i for i, l in enumerate(labels) if l == label]
            balanced_indices.extend(label_indices[:min_count])
        
        return dataset.select(balanced_indices)
    
    else:
        raise ValueError(f"Unknown balancing method: {method}")


def export_dataset(dataset: DatasetDict, output_dir: str, 
                  format: str = 'json') -> None:
    """Export dataset to files."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    for split_name, split_data in dataset.items():
        if format == 'json':
            output_file = os.path.join(output_dir, f"{split_name}.json")
            split_data.to_json(output_file)
        elif format == 'csv':
            output_file = os.path.join(output_dir, f"{split_name}.csv")
            split_data.to_csv(output_file, index=False)
        elif format == 'parquet':
            output_file = os.path.join(output_dir, f"{split_name}.parquet")
            split_data.to_parquet(output_file)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    logging.info(f"Dataset exported to {output_dir} in {format} format")


def create_data_splits(dataset: Dataset, train_ratio: float = 0.8, 
                      val_ratio: float = 0.1, test_ratio: float = 0.1,
                      label_column: str = 'label', seed: int = 42) -> DatasetDict:
    """Create train/validation/test splits from a single dataset."""
    
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Split ratios must sum to 1.0"
    
    # First split: train vs (val + test)
    train_test_split = dataset.train_test_split(
        test_size=val_ratio + test_ratio,
        seed=seed,
        stratify_by_column=label_column
    )
    
    # Second split: val vs test
    val_test_split = train_test_split['test'].train_test_split(
        test_size=test_ratio / (val_ratio + test_ratio),
        seed=seed,
        stratify_by_column=label_column
    )
    
    return DatasetDict({
        'train': train_test_split['train'],
        'validation': val_test_split['train'],
        'test': val_test_split['test']
    })
```

This configuration provides a comprehensive foundation for BERT fine-tuning experiments with extensive customization options and utility functions.
