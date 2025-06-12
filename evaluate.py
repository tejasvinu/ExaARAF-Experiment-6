import argparse
import json
import os
import logging
from typing import Dict, List, Any

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import wandb


def setup_logging() -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for evaluation."""
    parser = argparse.ArgumentParser(description="BERT Model Evaluation")
    
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the fine-tuned model")
    parser.add_argument("--dataset_name", type=str, default="imdb",
                       help="Dataset name for evaluation")
    parser.add_argument("--dataset_config", type=str, default=None,
                       help="Dataset configuration")
    parser.add_argument("--test_split", type=str, default="test",
                       help="Test split name")
    parser.add_argument("--text_column", type=str, default="text",
                       help="Text column name")
    parser.add_argument("--label_column", type=str, default="label",
                       help="Label column name")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size for evaluation")
    parser.add_argument("--max_length", type=int, default=512,
                       help="Maximum sequence length")
    parser.add_argument("--output_dir", type=str, default="./evaluation_results",
                       help="Output directory for results")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use")
    parser.add_argument("--save_predictions", type=bool, default=True,
                       help="Save detailed predictions")
    
    return parser.parse_args()


def load_model_and_tokenizer(model_path: str, device: str):
    """Load the fine-tuned model and tokenizer."""
    logging.info(f"Loading model from {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model.to(device)
    model.eval()
    
    return model, tokenizer, device


def evaluate_model(model, tokenizer, dataset, args, device: str) -> Dict[str, Any]:
    """Evaluate the model on the dataset."""
    logging.info("Starting model evaluation")
    
    predictions = []
    true_labels = []
    texts = []
    probabilities = []
    
    # Process in batches
    for i in range(0, len(dataset), args.batch_size):
        batch_texts = dataset[args.text_column][i:i+args.batch_size]
        batch_labels = dataset[args.label_column][i:i+args.batch_size]
        
        # Tokenize
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt"
        ).to(device)
        
        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)
        
        predictions.extend(preds.cpu().numpy())
        probabilities.extend(probs.cpu().numpy())
        true_labels.extend(batch_labels)
        texts.extend(batch_texts)
    
    return {
        'predictions': predictions,
        'true_labels': true_labels,
        'probabilities': probabilities,
        'texts': texts
    }


def compute_detailed_metrics(predictions: List[int], true_labels: List[int], label_names: List[str]) -> Dict[str, Any]:
    """Compute detailed evaluation metrics."""
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
    
    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, support = precision_recall_fscore_support(true_labels, predictions, average=None)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(true_labels, predictions, average='macro')
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(true_labels, predictions, average='weighted')
    
    # Classification report
    report = classification_report(true_labels, predictions, target_names=label_names, output_dict=True)
    
    # Confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    
    return {
        'accuracy': accuracy,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'weighted_f1': weighted_f1,
        'per_class_precision': precision.tolist(),
        'per_class_recall': recall.tolist(),
        'per_class_f1': f1.tolist(),
        'per_class_support': support.tolist(),
        'classification_report': report,
        'confusion_matrix': cm.tolist()
    }


def create_visualizations(results: Dict[str, Any], metrics: Dict[str, Any], 
                         label_names: List[str], output_dir: str) -> None:
    """Create and save visualizations."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Confusion Matrix
    plt.figure(figsize=(10, 8))
    cm = np.array(metrics['confusion_matrix'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_names, yticklabels=label_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Per-class metrics
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Precision
    axes[0].bar(label_names, metrics['per_class_precision'])
    axes[0].set_title('Precision per Class')
    axes[0].set_ylabel('Precision')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Recall
    axes[1].bar(label_names, metrics['per_class_recall'])
    axes[1].set_title('Recall per Class')
    axes[1].set_ylabel('Recall')
    axes[1].tick_params(axis='x', rotation=45)
    
    # F1-score
    axes[2].bar(label_names, metrics['per_class_f1'])
    axes[2].set_title('F1-Score per Class')
    axes[2].set_ylabel('F1-Score')
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'per_class_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Probability distribution
    probabilities = np.array(results['probabilities'])
    max_probs = np.max(probabilities, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.hist(max_probs, bins=50, alpha=0.7, edgecolor='black')
    plt.title('Distribution of Maximum Prediction Probabilities')
    plt.xlabel('Maximum Probability')
    plt.ylabel('Frequency')
    plt.axvline(np.mean(max_probs), color='red', linestyle='--', 
                label=f'Mean: {np.mean(max_probs):.3f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'probability_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()


def save_detailed_results(results: Dict[str, Any], metrics: Dict[str, Any], 
                         label_names: List[str], output_dir: str) -> None:
    """Save detailed evaluation results."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metrics
    with open(os.path.join(output_dir, 'detailed_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    
    # Save predictions with details
    if results.get('texts'):
        predictions_df = pd.DataFrame({
            'text': results['texts'],
            'true_label': [label_names[i] for i in results['true_labels']],
            'predicted_label': [label_names[i] for i in results['predictions']],
            'correct': [t == p for t, p in zip(results['true_labels'], results['predictions'])],
            'max_probability': [max(prob) for prob in results['probabilities']],
            'prediction_confidence': [prob[pred] for prob, pred in zip(results['probabilities'], results['predictions'])]
        })
        
        # Add probability columns for each class
        for i, label in enumerate(label_names):
            predictions_df[f'prob_{label}'] = [prob[i] for prob in results['probabilities']]
        
        predictions_df.to_csv(os.path.join(output_dir, 'detailed_predictions.csv'), index=False)
        
        # Save misclassified examples
        misclassified = predictions_df[~predictions_df['correct']].copy()
        misclassified = misclassified.sort_values('prediction_confidence', ascending=True)
        misclassified.to_csv(os.path.join(output_dir, 'misclassified_examples.csv'), index=False)
        
        # Save high-confidence correct predictions
        correct_high_conf = predictions_df[
            (predictions_df['correct']) & (predictions_df['prediction_confidence'] > 0.9)
        ].copy()
        correct_high_conf = correct_high_conf.sort_values('prediction_confidence', ascending=False)
        correct_high_conf.to_csv(os.path.join(output_dir, 'high_confidence_correct.csv'), index=False)


def main():
    """Main evaluation function."""
    args = parse_arguments()
    setup_logging()
    
    # Load model and tokenizer
    model, tokenizer, device = load_model_and_tokenizer(args.model_path, args.device)
    
    # Load dataset
    logging.info(f"Loading dataset: {args.dataset_name}")
    if args.dataset_config:
        dataset = load_dataset(args.dataset_name, args.dataset_config)
    else:
        dataset = load_dataset(args.dataset_name)
    
    test_dataset = dataset[args.test_split]
    
    # Get label names
    if hasattr(test_dataset.features[args.label_column], 'names'):
        label_names = test_dataset.features[args.label_column].names
    else:
        unique_labels = sorted(set(test_dataset[args.label_column]))
        label_names = [f"label_{i}" for i in unique_labels]
    
    logging.info(f"Label names: {label_names}")
    
    # Evaluate model
    results = evaluate_model(model, tokenizer, test_dataset, args, device)
    
    # Compute metrics
    metrics = compute_detailed_metrics(results['predictions'], results['true_labels'], label_names)
    
    # Print summary
    print("\n" + "="*50)
    print("EVALUATION RESULTS SUMMARY")
    print("="*50)
    print(f"Dataset: {args.dataset_name}")
    print(f"Test samples: {len(results['predictions'])}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
    print("\nPer-class metrics:")
    for i, label in enumerate(label_names):
        print(f"  {label}: P={metrics['per_class_precision'][i]:.4f}, "
              f"R={metrics['per_class_recall'][i]:.4f}, "
              f"F1={metrics['per_class_f1'][i]:.4f}, "
              f"Support={metrics['per_class_support'][i]}")
    
    # Create visualizations
    create_visualizations(results, metrics, label_names, args.output_dir)
    
    # Save detailed results
    if args.save_predictions:
        save_detailed_results(results, metrics, label_names, args.output_dir)
    
    logging.info(f"Evaluation completed. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
