"""Command-line entry point for the DiCo-NLI pipeline."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

from DataHandler import read_transform_dico
from EncoderTraining import MODEL_IDS, TrainingConfig, train_encoder
from Evaluator import append_result, evaluate, load_saved_model, print_report


def main():
    parser = argparse.ArgumentParser(description="Train a DiCo-NLI track 1 classifier")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--model", choices=MODEL_IDS, help="Train and evaluate a model")
    mode.add_argument("--evaluate-saved", type=int, metavar="ID", help="Evaluate the saved model for a result ID")
    parser.add_argument("--save", action="store_true", help="Save the trained model and tokenizer in saved_models")
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--precision", choices=("float32", "float16", "bfloat16"), default="float32")
    args = parser.parse_args()
    project_dir = Path(__file__).resolve().parent
    if args.evaluate_saved is not None:
        if args.save:
            parser.error("--save cannot be used with --evaluate-saved")
        paths = list((project_dir / "saved_models").glob(f"{args.evaluate_saved}_*"))
        if len(paths) != 1 or not paths[0].is_dir():
            parser.error(f"No unique saved model found for result ID {args.evaluate_saved}")
        model, tokenizer, settings = load_saved_model(paths[0])
        report = evaluate(model, tokenizer, max_length=settings["max_length"], batch_size=settings["batch_size"])
        print_report(report)
        return
    train_path = project_dir / "data/train.csv"
    if not train_path.is_file():
        parser.error(f"Cleaned training data not found at {train_path}. Run python PreProcessing.py first.")
    _, pairs, labels = read_transform_dico(train_path)
    val_path = project_dir / "data/val.csv"
    if not val_path.is_file():
        parser.error(f"Validation data not found at {val_path}. Run python PreProcessing.py first.")
    _, val_pairs, val_labels = read_transform_dico(val_path)
    config = TrainingConfig(model=args.model, epochs=args.epochs, batch_size=args.batch_size,
                            learning_rate=args.learning_rate, max_length=args.max_length,
                            gradient_accumulation_steps=args.gradient_accumulation_steps,
                            weight_decay=args.weight_decay, seed=args.seed,
                            precision=args.precision)
    model, tokenizer, loss_history = train_encoder(pairs, labels, val_pairs, val_labels, config)
    report = evaluate(model, tokenizer, max_length=config.max_length, batch_size=config.batch_size)
    result_id = append_result(report, asdict(config), loss_history)
    print(f"Saved result {result_id} to results.csv")
    if args.save:
        target = project_dir / "saved_models" / f"{result_id}_{args.model}"
        target.mkdir(parents=True, exist_ok=False)
        model.save_pretrained(target)
        tokenizer.save_pretrained(target)
        (target / "training_config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
        print(f"Saved model to {target}")


if __name__ == "__main__":
    main()
