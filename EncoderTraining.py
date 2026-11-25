"""Train supported encoder models for DiCo-NLI classification."""
from dataclasses import dataclass
from tempfile import TemporaryDirectory

import numpy as np
import torch
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, EarlyStoppingCallback, Trainer, TrainerCallback,
                          TrainingArguments)
from transformers.trainer_callback import PrinterCallback, ProgressCallback
from transformers.trainer_utils import enable_full_determinism, set_seed

from DataHandler import LABELS, LABEL_TO_ID


MODEL_IDS = {
    "deberta-v3-base": "microsoft/deberta-v3-base",
    "modernbert-base": "answerdotai/ModernBERT-base",
    "gte-en-mlm-base": "Alibaba-NLP/gte-en-mlm-base",
    "bert-base-uncased": "google-bert/bert-base-uncased",
}


class TrainingProgress(ProgressCallback):
    """Keep progress bars without printing metrics saved in results.csv."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        pass


class PruningCallback(TrainerCallback):
    """Report validation loss to Optuna after each epoch."""

    def __init__(self, trial):
        self.trial = trial

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        self.trial.report(metrics["eval_loss"], step=round(state.epoch))
        if self.trial.should_prune():
            from optuna import TrialPruned

            raise TrialPruned()


@dataclass
class TrainingConfig:
    model: str
    epochs: float = 3
    batch_size: int = 1
    learning_rate: float = 2e-5
    max_length: int = 256
    gradient_accumulation_steps: int = 1
    weight_decay: float = 0.0
    seed: int = 42
    precision: str = "float32"
    patience: int = 3
    deterministic: bool = False


class PairDataset(torch.utils.data.Dataset):
    def __init__(self, pairs, labels, tokenizer, max_length):
        if labels is None or len(pairs) == 0 or len(pairs) != len(labels):
            raise ValueError("Training requires nonempty text pairs and one label per pair")
        if any(not isinstance(label, (int, np.integer)) or not 0 <= label < len(LABELS) for label in labels):
            raise ValueError("Training labels must be integer IDs from DataHandler")
        self.pairs, self.labels, self.tokenizer, self.max_length = pairs, labels, tokenizer, max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        first, second = self.pairs[index]
        tokens = self.tokenizer(str(first), str(second), truncation=True, max_length=self.max_length)
        tokens["labels"] = int(self.labels[index])
        return tokens


def train_encoder(pairs, labels, val_pairs, val_labels, config: TrainingConfig, *, trial=None):
    """Return the best model, tokenizer, loss history and best epoch."""
    if config.model not in MODEL_IDS:
        raise ValueError(f"Unknown model {config.model!r}; choose from {', '.join(MODEL_IDS)}")
    if config.epochs <= 0 or config.batch_size < 1 or config.max_length < 2 or config.gradient_accumulation_steps < 1 or config.learning_rate <= 0 or config.weight_decay < 0:
        raise ValueError("Invalid training hyperparameters")
    if config.patience < 1:
        raise ValueError("patience must be a positive number of epochs")
    if config.precision not in {"float32", "float16", "bfloat16"}:
        raise ValueError("precision must be float32, float16, or bfloat16")
    if config.precision != "float32" and not torch.cuda.is_available():
        raise ValueError("Mixed precision requires CUDA")
    # Validate before loading large model weights.
    if labels is None or len(pairs) == 0 or len(pairs) != len(labels):
        raise ValueError("Training requires nonempty text pairs and one label per pair")
    if config.deterministic:
        enable_full_determinism(config.seed)
    else:
        set_seed(config.seed)
    model_id = MODEL_IDS[config.model]
    trust_remote_code = config.model == "gte-en-mlm-base"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, num_labels=len(LABELS), id2label=dict(enumerate(LABELS)), label2id=LABEL_TO_ID,
        problem_type="single_label_classification", trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    dataset = PairDataset(pairs, labels, tokenizer, config.max_length)
    val_dataset = PairDataset(val_pairs, val_labels, tokenizer, config.max_length)
    # Checkpoints are temporary; the best weights are loaded before cleanup.
    with TemporaryDirectory(prefix="dico_training_") as temporary_dir:
        arguments = TrainingArguments(
            output_dir=temporary_dir, num_train_epochs=config.epochs,
            per_device_train_batch_size=config.batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate, weight_decay=config.weight_decay,
            seed=config.seed, data_seed=config.seed, full_determinism=config.deterministic,
            fp16=config.precision == "float16", bf16=config.precision == "bfloat16",
            save_strategy="epoch", save_total_limit=2, report_to="none", logging_strategy="epoch",
            eval_strategy="epoch", per_device_eval_batch_size=config.batch_size,
            load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
        )
        trainer = Trainer(model=model, args=arguments, train_dataset=dataset, eval_dataset=val_dataset,
                          data_collator=DataCollatorWithPadding(tokenizer), processing_class=tokenizer)
        trainer.remove_callback(ProgressCallback)
        trainer.remove_callback(PrinterCallback)
        trainer.add_callback(TrainingProgress())
        if trial is None:
            trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=config.patience))
        else:
            trainer.add_callback(PruningCallback(trial))
        trainer.train()
        train_logs = [entry for entry in trainer.state.log_history if "loss" in entry]
        eval_logs = [entry for entry in trainer.state.log_history if "eval_loss" in entry]
        if len(train_logs) != len(eval_logs) or not train_logs:
            raise RuntimeError("Could not match training and validation loss logs")
        if not trainer.state.best_model_checkpoint:
            raise RuntimeError("Training did not produce a best checkpoint")
        best_step = int(trainer.state.best_model_checkpoint.rsplit("-", 1)[-1])
        best_logs = [entry for entry in eval_logs if entry["step"] == best_step]
        if len(best_logs) != 1:
            raise RuntimeError("Could not identify the best checkpoint epoch")
        best_epoch = best_logs[0]["epoch"]
        loss_history = []
        for train_log, eval_log in zip(train_logs, eval_logs):
            if abs(train_log["epoch"] - eval_log["epoch"]) > 1e-6:
                raise RuntimeError("Training and validation losses have different epoch values")
            loss_history.append({"epoch": train_log["epoch"], "train_loss": train_log["loss"],
                                 "val_loss": eval_log["eval_loss"]})
    return trainer.model, tokenizer, loss_history, best_epoch
