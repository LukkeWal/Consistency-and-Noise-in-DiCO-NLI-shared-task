"""Command-line entry point for the DiCo-NLI pipeline."""
import argparse
from dataclasses import asdict, replace
import gc
import json
from pathlib import Path
import statistics

from DataHandler import read_transform_dico
from EncoderTraining import MODEL_IDS, TrainingConfig, train_encoder
from Evaluator import append_result, evaluate, load_saved_model, print_report

# Run i (counting from 0) uses seed BASE_SEED + i.
BASE_SEED = 42
SUMMARY_METRICS = ("weighted_f1", "soft_cons", "hard_cons", "accuracy")

def print_seed_summary(runs):
    """Print per-seed metrics and mean ± sample standard deviation."""
    print(f"\nSummary over {len(runs)} seeds")
    print(f"{'id':>4} {'seed':>6} " + " ".join(f"{m:>11}" for m in SUMMARY_METRICS))
    for run in runs:
        print(f"{run['id']:>4} {run['seed']:>6} " + " ".join(f"{run[m]:>11.4f}" for m in SUMMARY_METRICS))
    for name, func in (("mean", statistics.mean), ("std", statistics.stdev)):
        print(f"{name:>11} " + " ".join(f"{func([run[m] for run in runs]):>11.4f}" for m in SUMMARY_METRICS))

def tune_learning_rate(pairs, labels, val_pairs, val_labels, config, n_trials):
    """Select the learning rate with the lowest validation loss."""
    import optuna

    def objective(trial):
        learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True)
        trial_config = replace(config, learning_rate=learning_rate)
        model, tokenizer, loss_history, _ = train_encoder(
            pairs, labels, val_pairs, val_labels, trial_config, trial=trial,
        )
        del model, tokenizer
        return min(row["val_loss"] for row in loss_history)

    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.HyperbandPruner(
            min_resource=1, max_resource=int(config.epochs), reduction_factor=3,
        ),
    )
    study.optimize(objective, n_trials=n_trials, gc_after_trial=True)
    learning_rate = study.best_params["learning_rate"]
    print(f"Selected learning rate {learning_rate:.6g} (validation loss {study.best_value:.4f})")
    return learning_rate


def main():
    parser = argparse.ArgumentParser(description="Train a DiCo-NLI track 1 classifier")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--model", choices=MODEL_IDS, help="Train and evaluate a model")
    mode.add_argument("--evaluate-saved", type=int, metavar="ID", help="Evaluate the saved model for a result ID")
    parser.add_argument("--save", action="store_true", help="Save the trained model and tokenizer in saved_models")
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--patience", type=int, default=3, help="Stop after this many epochs without validation loss improvement (default: 3)")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--tune-trials", type=int, metavar="N", default=0,
                        help="Tune the learning rate with N Optuna Hyperband trials")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=BASE_SEED,
                        help=f"Random seed for reproducibility (default: {BASE_SEED})")
    parser.add_argument("--num-seeds", type=int, default=1, metavar="N",
                        help="Number of runs; run i uses seed + i (default: 1)")
    parser.add_argument("--deterministic", action="store_true",
                        help="Force deterministic CUDA kernels for bit-for-bit reruns (slower)")
    parser.add_argument("--precision", choices=("float32", "float16", "bfloat16"), default="float32")
    args = parser.parse_args()
    if args.num_seeds < 1:
        parser.error("--num-seeds must be at least 1")
    if args.tune_trials < 0:
        parser.error("--tune-trials must be nonnegative")
    if args.tune_trials:
        if args.model is None or args.num_seeds != 1:
            parser.error("--tune-trials requires --model and --num-seeds 1")
        if not args.epochs.is_integer() or args.epochs < 1:
            parser.error("--tune-trials requires a positive integer --epochs")
    seed_list = [BASE_SEED + i for i in range(args.num_seeds)]
    project_dir = Path(__file__).resolve().parent
    if args.evaluate_saved is not None:
        if args.save:
            parser.error("--save cannot be used with --evaluate-saved")
        if args.num_seeds != 1 or args.deterministic:
            parser.error("Seed options cannot be used with --evaluate-saved")
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
    group_id, runs = None, []
    for index, seed in enumerate(seed_list, start=1):
        if len(seed_list) > 1:
            print(f"\n=== Run {index}/{len(seed_list)} with seed {seed} ===")
        config = TrainingConfig(model=args.model, epochs=args.epochs, batch_size=args.batch_size,
                                learning_rate=args.learning_rate, max_length=args.max_length,
                                gradient_accumulation_steps=args.gradient_accumulation_steps,
                                weight_decay=args.weight_decay, seed=seed,
                                precision=args.precision, patience=args.patience,
                                deterministic=args.deterministic)
        if args.tune_trials:
            learning_rate = tune_learning_rate(pairs, labels, val_pairs, val_labels, config, args.tune_trials)
            config = replace(config, learning_rate=learning_rate)
        model, tokenizer, loss_history, best_epoch = train_encoder(pairs, labels, val_pairs, val_labels, config)
        report = evaluate(model, tokenizer, max_length=config.max_length, batch_size=config.batch_size)
        # All runs from one call share the first run's ID as group_id.
        result_id = append_result(report, asdict(config), loss_history, best_epoch, group_id=group_id)
        group_id = group_id or result_id
        print(f"Saved result {result_id} (group {group_id}, seed {seed}) to results.csv")
        runs.append({"id": result_id, "seed": seed, "weighted_f1": report.weighted_f1,
                     "soft_cons": report.soft_cons, "hard_cons": report.hard_cons,
                     "accuracy": report.classification.accuracy})
        if args.save:
            target = project_dir / "saved_models" / f"{result_id}_{args.model}_seed{seed}"
            target.mkdir(parents=True, exist_ok=False)
            model.save_pretrained(target)
            tokenizer.save_pretrained(target)
            (target / "training_config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
            print(f"Saved model to {target}")
        # Free GPU memory before the next seed loads a fresh model.
        del model, tokenizer
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
    if len(runs) > 1:
        print_seed_summary(runs)


if __name__ == "__main__":
    main()
