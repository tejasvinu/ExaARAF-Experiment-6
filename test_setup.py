#!/usr/bin/env python3
"""
Quick test script to validate the BERT fine-tuning setup
"""

import os
import sys
import argparse
import logging
from pathlib import Path

def test_imports():
    """Test if all required packages can be imported."""
    print("Testing imports...")
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name()}")
        else:
            print("⚠ CUDA not available, will use CPU")
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        return False
    
    try:
        import transformers
        print(f"✓ Transformers {transformers.__version__}")
    except ImportError as e:
        print(f"✗ Transformers import failed: {e}")
        return False
    
    try:
        import datasets
        print(f"✓ Datasets {datasets.__version__}")
    except ImportError as e:
        print(f"✗ Datasets import failed: {e}")
        return False
    
    try:
        import sklearn
        print(f"✓ Scikit-learn {sklearn.__version__}")
    except ImportError as e:
        print(f"✗ Scikit-learn import failed: {e}")
        return False
    
    optional_packages = {
        'wandb': 'Weights & Biases',
        'matplotlib': 'Matplotlib',
        'seaborn': 'Seaborn',
        'pandas': 'Pandas'
    }
    
    for package, name in optional_packages.items():
        try:
            module = __import__(package)
            version = getattr(module, '__version__', 'unknown')
            print(f"✓ {name} {version}")
        except ImportError:
            print(f"⚠ {name} not available (optional)")
    
    return True


def test_quick_training():
    """Run a quick training test with minimal data."""
    print("\nTesting quick training...")
    
    try:
        # Import our training script components
        sys.path.append(os.path.dirname(__file__))
        from train import parse_arguments, load_and_prepare_dataset
        
        # Create test arguments
        test_args = argparse.Namespace(
            dataset_name="imdb",
            dataset_config=None,
            text_column="text",
            label_column="label",
            train_split="train",
            validation_split="test",
            test_split=None,
            max_samples=100,  # Very small for testing
            validation_ratio=0.2,
            model_name="distilbert-base-uncased",  # Smaller model
            num_labels=2,
            max_length=128,  # Shorter sequences
            truncation=True,
            padding="max_length",
            output_dir="./quick_test",
            num_train_epochs=1,  # Just 1 epoch
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            learning_rate=2e-5,
            weight_decay=0.01,
            warmup_steps=10,
            warmup_ratio=0.0,
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_epsilon=1e-8,
            max_grad_norm=1.0,
            lr_scheduler_type="linear",
            evaluation_strategy="epoch",
            eval_steps=50,
            save_strategy="epoch",
            save_steps=50,
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="eval_accuracy",
            greater_is_better=True,
            early_stopping_patience=1,
            early_stopping_threshold=0.0,
            logging_dir="./quick_test/logs",
            logging_steps=10,
            log_level="INFO",
            report_to="none",
            wandb_project="test",
            wandb_entity=None,
            wandb_run_name=None,
            device="auto",
            fp16=False,
            bf16=False,
            dataloader_num_workers=0,
            seed=42,
            resume_from_checkpoint=None,
            ignore_data_skip=False,
            run_name="quick_test",
            do_train=True,
            do_eval=True,
            do_predict=False
        )
        
        print("✓ Arguments parsed successfully")
        
        # Test dataset loading
        try:
            dataset, label_names = load_and_prepare_dataset(test_args)
            print(f"✓ Dataset loaded: {len(dataset['train'])} train, {len(dataset['test'])} test samples")
            print(f"✓ Labels: {label_names}")
        except Exception as e:
            print(f"✗ Dataset loading failed: {e}")
            return False
        
        print("✓ Quick test components working correctly")
        return True
        
    except Exception as e:
        print(f"✗ Quick training test failed: {e}")
        return False


def test_model_loading():
    """Test loading a pre-trained model."""
    print("\nTesting model loading...")
    
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        
        model_name = "distilbert-base-uncased"
        print(f"Loading {model_name}...")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=2
        )
        
        print(f"✓ Model loaded: {model.config.model_type}")
        print(f"✓ Tokenizer loaded: {len(tokenizer)} tokens in vocabulary")
        
        # Test tokenization
        test_text = "This is a test sentence."
        tokens = tokenizer(test_text, return_tensors="pt", truncation=True, padding=True)
        print(f"✓ Tokenization works: {tokens['input_ids'].shape}")
        
        # Test model forward pass
        with torch.no_grad():
            outputs = model(**tokens)
            logits = outputs.logits
            print(f"✓ Model forward pass works: {logits.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Model loading test failed: {e}")
        return False


def cleanup_test_files():
    """Clean up test files."""
    import shutil
    
    test_dirs = ["./quick_test", "./test_results"]
    
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            try:
                shutil.rmtree(test_dir)
                print(f"✓ Cleaned up {test_dir}")
            except Exception as e:
                print(f"⚠ Could not clean up {test_dir}: {e}")


def main():
    """Main test function."""
    print("=" * 60)
    print("BERT FINE-TUNING SETUP VALIDATION")
    print("=" * 60)
    
    all_tests_passed = True
    
    # Test imports
    if not test_imports():
        all_tests_passed = False
    
    # Test model loading
    if not test_model_loading():
        all_tests_passed = False
    
    # Test training components
    if not test_quick_training():
        all_tests_passed = False
    
    print("\n" + "=" * 60)
    if all_tests_passed:
        print("🎉 ALL TESTS PASSED! Setup is ready for BERT fine-tuning.")
        print("\nNext steps:")
        print("1. Run a quick training test: python train.py --max_samples 100 --num_train_epochs 1")
        print("2. Check the example scripts in the 'examples/' directory")
        print("3. Review the comprehensive README.md for detailed usage")
    else:
        print("❌ SOME TESTS FAILED! Please check the errors above.")
        print("\nTroubleshooting:")
        print("1. Make sure all dependencies are installed: pip install -r requirements.txt")
        print("2. Check your Python environment and package versions")
        print("3. Ensure you have internet connection for downloading models/datasets")
    
    print("=" * 60)
    
    # Cleanup
    cleanup_test_files()
    
    return all_tests_passed


if __name__ == "__main__":
    # Set up basic logging
    logging.basicConfig(level=logging.WARNING)  # Reduce noise during testing
    
    success = main()
    sys.exit(0 if success else 1)
