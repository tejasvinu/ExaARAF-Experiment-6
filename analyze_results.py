import os
import json
import argparse
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Analyze BERT fine-tuning experiment results")
    
    parser.add_argument("--results_dir", type=str, required=True,
                       help="Directory containing experiment results")
    parser.add_argument("--output_dir", type=str, default="./analysis_results",
                       help="Output directory for analysis")
    
    return parser.parse_args()


def load_training_logs(results_dir: str) -> Dict[str, Any]:
    """Load training logs and metrics."""
    logs_dir = os.path.join(results_dir, "logs")
    
    # Try to find trainer_state.json
    trainer_state_path = os.path.join(results_dir, "trainer_state.json")
    training_logs = {}
    
    if os.path.exists(trainer_state_path):
        with open(trainer_state_path, 'r') as f:
            trainer_state = json.load(f)
        
        # Extract training history
        if 'log_history' in trainer_state:
            training_logs['log_history'] = trainer_state['log_history']
        
        if 'best_metric' in trainer_state:
            training_logs['best_metric'] = trainer_state['best_metric']
        
        if 'best_model_checkpoint' in trainer_state:
            training_logs['best_model_checkpoint'] = trainer_state['best_model_checkpoint']
    
    return training_logs


def load_evaluation_results(results_dir: str) -> Dict[str, Any]:
    """Load evaluation results."""
    eval_results = {}
    
    # Load eval_results.json
    eval_results_path = os.path.join(results_dir, "eval_results.json")
    if os.path.exists(eval_results_path):
        with open(eval_results_path, 'r') as f:
            eval_results['final_eval'] = json.load(f)
    
    # Load test predictions if available
    test_predictions_path = os.path.join(results_dir, "test_predictions.json")
    if os.path.exists(test_predictions_path):
        with open(test_predictions_path, 'r') as f:
            eval_results['test_predictions'] = json.load(f)
    
    return eval_results


def load_experiment_config(results_dir: str) -> Dict[str, Any]:
    """Load experiment configuration."""
    config_path = os.path.join(results_dir, "experiment_config.json")
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    
    return {}


def create_training_plots(training_logs: Dict[str, Any], output_dir: str) -> None:
    """Create training progress plots."""
    if 'log_history' not in training_logs:
        print("No training log history found")
        return
    
    log_history = training_logs['log_history']
    
    # Extract metrics
    steps = []
    train_loss = []
    eval_loss = []
    eval_accuracy = []
    eval_f1 = []
    learning_rates = []
    
    for entry in log_history:
        if 'step' in entry:
            steps.append(entry['step'])
        
        if 'train_loss' in entry:
            train_loss.append(entry['train_loss'])
        
        if 'eval_loss' in entry:
            eval_loss.append(entry['eval_loss'])
        
        if 'eval_accuracy' in entry:
            eval_accuracy.append(entry['eval_accuracy'])
        
        if 'eval_f1' in entry:
            eval_f1.append(entry['eval_f1'])
        
        if 'learning_rate' in entry:
            learning_rates.append(entry['learning_rate'])
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Training and validation loss
    if train_loss and eval_loss:
        eval_steps = [step for step, entry in zip(steps, log_history) if 'eval_loss' in entry]
        axes[0, 0].plot(steps[:len(train_loss)], train_loss, label='Training Loss', alpha=0.7)
        axes[0, 0].plot(eval_steps, eval_loss, label='Validation Loss', marker='o')
        axes[0, 0].set_xlabel('Steps')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy over time
    if eval_accuracy:
        eval_steps = [step for step, entry in zip(steps, log_history) if 'eval_accuracy' in entry]
        axes[0, 1].plot(eval_steps, eval_accuracy, label='Validation Accuracy', marker='o', color='green')
        axes[0, 1].set_xlabel('Steps')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Validation Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # F1 score over time
    if eval_f1:
        eval_steps = [step for step, entry in zip(steps, log_history) if 'eval_f1' in entry]
        axes[1, 0].plot(eval_steps, eval_f1, label='Validation F1', marker='o', color='orange')
        axes[1, 0].set_xlabel('Steps')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].set_title('Validation F1 Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Learning rate schedule
    if learning_rates:
        lr_steps = [step for step, entry in zip(steps, log_history) if 'learning_rate' in entry]
        axes[1, 1].plot(lr_steps, learning_rates, label='Learning Rate', color='red')
        axes[1, 1].set_xlabel('Steps')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_progress.png'), dpi=300, bbox_inches='tight')
    plt.close()


def create_metrics_summary(eval_results: Dict[str, Any], config: Dict[str, Any], output_dir: str) -> None:
    """Create a comprehensive metrics summary."""
    summary = {
        'experiment_config': {
            'model_name': config.get('model_name', 'Unknown'),
            'dataset_name': config.get('dataset_name', 'Unknown'),
            'num_epochs': config.get('num_train_epochs', 'Unknown'),
            'batch_size': config.get('per_device_train_batch_size', 'Unknown'),
            'learning_rate': config.get('learning_rate', 'Unknown'),
            'max_length': config.get('max_length', 'Unknown'),
            'seed': config.get('seed', 'Unknown')
        }
    }
    
    # Final evaluation metrics
    if 'final_eval' in eval_results:
        summary['final_evaluation'] = eval_results['final_eval']
    
    # Test set performance
    if 'test_predictions' in eval_results and 'metrics' in eval_results['test_predictions']:
        summary['test_performance'] = eval_results['test_predictions']['metrics']
    
    # Save summary
    with open(os.path.join(output_dir, 'experiment_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    # Create a formatted report
    report_lines = []
    report_lines.append("BERT FINE-TUNING EXPERIMENT SUMMARY")
    report_lines.append("=" * 50)
    report_lines.append("")
    
    # Configuration
    report_lines.append("EXPERIMENT CONFIGURATION:")
    for key, value in summary['experiment_config'].items():
        report_lines.append(f"  {key.replace('_', ' ').title()}: {value}")
    report_lines.append("")
    
    # Final evaluation
    if 'final_evaluation' in summary:
        report_lines.append("VALIDATION SET PERFORMANCE:")
        for key, value in summary['final_evaluation'].items():
            if isinstance(value, float):
                report_lines.append(f"  {key.replace('_', ' ').title()}: {value:.4f}")
            else:
                report_lines.append(f"  {key.replace('_', ' ').title()}: {value}")
        report_lines.append("")
    
    # Test performance
    if 'test_performance' in summary:
        report_lines.append("TEST SET PERFORMANCE:")
        for key, value in summary['test_performance'].items():
            if isinstance(value, float):
                report_lines.append(f"  {key.replace('_', ' ').title()}: {value:.4f}")
            else:
                report_lines.append(f"  {key.replace('_', ' ').title()}: {value}")
        report_lines.append("")
    
    # Save report
    with open(os.path.join(output_dir, 'experiment_report.txt'), 'w') as f:
        f.write('\n'.join(report_lines))


def analyze_checkpoints(results_dir: str, output_dir: str) -> None:
    """Analyze model checkpoints if available."""
    checkpoints = []
    
    # Find all checkpoint directories
    for item in os.listdir(results_dir):
        if item.startswith('checkpoint-'):
            checkpoint_path = os.path.join(results_dir, item)
            if os.path.isdir(checkpoint_path):
                # Extract step number
                step_num = int(item.split('-')[1])
                checkpoints.append({
                    'step': step_num,
                    'path': checkpoint_path,
                    'name': item
                })
    
    if checkpoints:
        checkpoints.sort(key=lambda x: x['step'])
        
        checkpoint_info = {
            'total_checkpoints': len(checkpoints),
            'checkpoint_steps': [cp['step'] for cp in checkpoints],
            'checkpoint_names': [cp['name'] for cp in checkpoints]
        }
        
        with open(os.path.join(output_dir, 'checkpoint_info.json'), 'w') as f:
            json.dump(checkpoint_info, f, indent=2)
        
        print(f"Found {len(checkpoints)} checkpoints")


def main():
    """Main analysis function."""
    args = parse_arguments()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Analyzing experiment results from: {args.results_dir}")
    
    # Load all data
    training_logs = load_training_logs(args.results_dir)
    eval_results = load_evaluation_results(args.results_dir)
    config = load_experiment_config(args.results_dir)
    
    # Create visualizations
    if training_logs:
        print("Creating training progress plots...")
        create_training_plots(training_logs, args.output_dir)
    
    # Create metrics summary
    print("Creating metrics summary...")
    create_metrics_summary(eval_results, config, args.output_dir)
    
    # Analyze checkpoints
    print("Analyzing checkpoints...")
    analyze_checkpoints(args.results_dir, args.output_dir)
    
    print(f"Analysis completed. Results saved to: {args.output_dir}")
    
    # Print summary to console
    summary_path = os.path.join(args.output_dir, 'experiment_report.txt')
    if os.path.exists(summary_path):
        print("\nEXPERIMENT SUMMARY:")
        print("-" * 50)
        with open(summary_path, 'r') as f:
            print(f.read())


if __name__ == "__main__":
    main()
