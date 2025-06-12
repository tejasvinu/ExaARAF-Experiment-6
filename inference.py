import argparse
import json
import logging
import os
from typing import Dict, List, Any, Optional

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline


def setup_logging() -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for inference."""
    parser = argparse.ArgumentParser(description="BERT Model Inference")
    
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the fine-tuned model")
    parser.add_argument("--input_text", type=str, default=None,
                       help="Single text to classify")
    parser.add_argument("--input_file", type=str, default=None,
                       help="File containing texts to classify (one per line)")
    parser.add_argument("--output_file", type=str, default=None,
                       help="Output file for predictions")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size for inference")
    parser.add_argument("--max_length", type=int, default=512,
                       help="Maximum sequence length")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use for inference")
    parser.add_argument("--return_probabilities", type=bool, default=True,
                       help="Return prediction probabilities")
    parser.add_argument("--top_k", type=int, default=None,
                       help="Return top-k predictions")
    
    return parser.parse_args()


def load_texts_from_file(filepath: str) -> List[str]:
    """Load texts from a file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f if line.strip()]
    return texts


def save_predictions(predictions: List[Dict[str, Any]], output_file: str) -> None:
    """Save predictions to a file."""
    if output_file.endswith('.json'):
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(predictions, f, indent=2, ensure_ascii=False)
    else:
        # Save as CSV
        import pandas as pd
        df = pd.DataFrame(predictions)
        df.to_csv(output_file, index=False)


def run_inference(model_path: str, texts: List[str], args: argparse.Namespace) -> List[Dict[str, Any]]:
    """Run inference on a list of texts."""
    logging.info(f"Loading model from {model_path}")
    
    # Determine device
    if args.device == "auto":
        device = 0 if torch.cuda.is_available() else -1
    elif args.device == "cuda":
        device = 0
    else:
        device = -1
    
    # Create classification pipeline
    classifier = pipeline(
        "text-classification",
        model=model_path,
        tokenizer=model_path,
        device=device,
        return_all_scores=args.return_probabilities,
        truncation=True,
        max_length=args.max_length
    )
    
    logging.info(f"Running inference on {len(texts)} texts")
    
    # Run predictions in batches
    all_predictions = []
    
    for i in range(0, len(texts), args.batch_size):
        batch_texts = texts[i:i+args.batch_size]
        batch_predictions = classifier(batch_texts)
        
        # Process batch results
        for j, (text, pred) in enumerate(zip(batch_texts, batch_predictions)):
            if args.return_probabilities:
                # Sort by score if returning all scores
                if isinstance(pred, list):
                    pred = sorted(pred, key=lambda x: x['score'], reverse=True)
                    if args.top_k:
                        pred = pred[:args.top_k]
                
                result = {
                    'text': text,
                    'predictions': pred,
                    'top_prediction': pred[0] if isinstance(pred, list) else pred
                }
            else:
                result = {
                    'text': text,
                    'prediction': pred
                }
            
            all_predictions.append(result)
        
        logging.info(f"Processed {min(i + args.batch_size, len(texts))}/{len(texts)} texts")
    
    return all_predictions


def print_predictions(predictions: List[Dict[str, Any]], args: argparse.Namespace) -> None:
    """Print predictions in a formatted way."""
    print("\n" + "="*80)
    print("PREDICTION RESULTS")
    print("="*80)
    
    for i, pred in enumerate(predictions):
        print(f"\nText {i+1}:")
        print(f"Input: {pred['text'][:100]}{'...' if len(pred['text']) > 100 else ''}")
        
        if args.return_probabilities and 'predictions' in pred:
            print("Predictions:")
            for p in pred['predictions']:
                print(f"  {p['label']}: {p['score']:.4f}")
        else:
            p = pred.get('prediction', pred.get('top_prediction', {}))
            print(f"Prediction: {p['label']} (confidence: {p['score']:.4f})")
        
        print("-" * 40)


def main():
    """Main inference function."""
    args = parse_arguments()
    setup_logging()
    
    # Validate arguments
    if not args.input_text and not args.input_file:
        raise ValueError("Either --input_text or --input_file must be provided")
    
    # Collect texts to classify
    texts = []
    if args.input_text:
        texts.append(args.input_text)
    
    if args.input_file:
        file_texts = load_texts_from_file(args.input_file)
        texts.extend(file_texts)
    
    if not texts:
        raise ValueError("No texts to classify")
    
    logging.info(f"Found {len(texts)} texts to classify")
    
    # Run inference
    predictions = run_inference(args.model_path, texts, args)
    
    # Print results
    print_predictions(predictions, args)
    
    # Save results if output file specified
    if args.output_file:
        save_predictions(predictions, args.output_file)
        logging.info(f"Predictions saved to {args.output_file}")
    
    logging.info("Inference completed successfully!")


if __name__ == "__main__":
    main()
