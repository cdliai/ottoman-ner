#!/usr/bin/env python3
"""
Ottoman NER Command Line Interface

Simple and unified CLI for Ottoman Turkish Named Entity Recognition.
"""

import argparse
import sys
import logging
import json
from pathlib import Path


# Import core functionality
from .core import OttomanNER
from .utils import setup_logging

logger = setup_logging(level=logging.INFO)


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser."""
    parser = argparse.ArgumentParser(
        prog="ottoman-ner",
        description="Ottoman Turkish Named Entity Recognition Toolkit",
        epilog="""
Examples:
  # Train a model
  ottoman-ner train --config configs/training.json
  
  # Evaluate a model
  ottoman-ner eval --model-path models/my-model --test-file data/test.conll
  
  # Make predictions
  ottoman-ner predict --model-path models/my-model --text "Sultan Abdülhamid"
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    subparsers = parser.add_subparsers(
        dest="command",
        help="Available commands",
        metavar="{train,eval,predict}"
    )
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train a NER model")
    train_parser.add_argument(
        "--config", "-c",
        type=str,
        required=True,
        help="Path to training configuration file"
    )
    
    # Eval command
    eval_parser = subparsers.add_parser("eval", help="Evaluate a NER model")
    eval_parser.add_argument(
        "--model-path", "-m",
        type=str,
        required=True,
        help="Path to trained model"
    )
    eval_parser.add_argument(
        "--test-file", "-t",
        type=str,
        required=True,
        help="Path to test data file"
    )
    eval_parser.add_argument(
        "--output-dir", "-o",
        type=str,
        help="Output directory for results"
    )
    
    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Make predictions")
    predict_parser.add_argument(
        "--model-path", "-m",
        type=str,
        help="Path to trained model (defaults to the published Hugging Face model)"
    )
    
    # Text input options (mutually exclusive)
    text_group = predict_parser.add_mutually_exclusive_group(required=True)
    text_group.add_argument(
        "--text",
        type=str,
        help="Text to analyze"
    )
    text_group.add_argument(
        "--input-file",
        type=str,
        help="Input file with texts to analyze"
    )
    
    predict_parser.add_argument(
        "--output-file",
        type=str,
        help="Output file for predictions (JSON format)"
    )
    
    return parser


def handle_train_command(args):
    """Handle training command."""
    logger.info(f"🚀 Starting training with config: {args.config}")
    
    try:
        # Load configuration
        config_path = Path(args.config)
        if not config_path.exists():
            logger.error(f"❌ Configuration file not found: {args.config}")
            return 1
            
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Initialize Ottoman NER
        ner = OttomanNER()
        
        # Train model
        results = ner.train_from_config(config)
        
        logger.info("✅ Training completed successfully!")
        final_f1 = results.get('eval_f1')
        if isinstance(final_f1, (int, float)):
            logger.info(f"📊 Final F1 Score: {final_f1:.4f}")
        else:
            logger.info(f"📊 Final F1 Score: {final_f1 if final_f1 is not None else 'N/A'}")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Training failed: {str(e)}")
        return 1


def handle_eval_command(args):
    """Handle evaluation command."""
    logger.info(f"📊 Evaluating model: {args.model_path}")
    
    try:
        # Initialize Ottoman NER
        ner = OttomanNER()
        
        # Evaluate model
        results = ner.evaluate(
            model_path=args.model_path,
            test_file=args.test_file,
            output_dir=args.output_dir
        )
        
        # Display results
        logger.info("✅ Evaluation completed!")
        logger.info(f"📊 Results:")
        logger.info(f"   Overall F1: {results['overall_f1']:.4f}")
        logger.info(f"   Precision:  {results['overall_precision']:.4f}")
        logger.info(f"   Recall:     {results['overall_recall']:.4f}")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {str(e)}")
        return 1


def handle_predict_command(args):
    """Handle prediction command."""
    try:
        # Initialize Ottoman NER
        ner = OttomanNER()
        
        # Load model
        target_model = args.model_path or ner.default_model_path
        logger.info(f"🔮 Loading model: {target_model}")
        ner.load_model(args.model_path)
        
        # Get texts to process
        if args.text:
            texts = [args.text]
        else:
            # Read from file
            input_path = Path(args.input_file)
            if not input_path.exists():
                logger.error(f"❌ Input file not found: {args.input_file}")
                return 1
            texts = input_path.read_text(encoding='utf-8').strip().split('\n')
        
        # Make predictions
        logger.info("🔍 Making predictions...")
        all_predictions = []
        
        for text in texts:
            if text.strip():
                entities = ner.predict(text)
                all_predictions.append({
                    'text': text,
                    'entities': entities
                })
        
        # Output results
        if args.output_file:
            output_path = Path(args.output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(all_predictions, f, ensure_ascii=False, indent=2)
            logger.info(f"💾 Predictions saved to: {args.output_file}")
        else:
            # Print to console
            for pred in all_predictions:
                print(f"\nText: {pred['text']}")
                if pred['entities']:
                    print("Entities:")
                    for entity in pred['entities']:
                        print(f"  - {entity['text']} ({entity['label']}) [{entity['start']}:{entity['end']}]")
                else:
                    print("  No entities found")
        
        logger.info("✅ Prediction completed!")
        return 0
        
    except Exception as e:
        logger.error(f"❌ Prediction failed: {str(e)}")
        return 1


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Handle commands
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        if args.command == "train":
            return handle_train_command(args)
        elif args.command == "eval":
            return handle_eval_command(args)
        elif args.command == "predict":
            return handle_predict_command(args)
        else:
            logger.error(f"❌ Unknown command: {args.command}")
            return 1
            
    except KeyboardInterrupt:
        logger.info("⏹️  Operation cancelled by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Command failed: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
