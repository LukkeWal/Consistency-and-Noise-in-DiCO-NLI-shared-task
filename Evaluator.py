"""Evaluate a classifier with the original DiCo-NLI scorer."""
import csv
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from DataHandler import LABELS, read_transform_dico
from SemEval_source.evaluation_functions.io import read_reference_file
from SemEval_source.evaluation_functions.metrics import compute_evaluation
from SemEval_source.evaluation_functions.schemas import PredictionRecord
from SemEval_source.evaluation_functions.validation import validate_prediction_records, validate_reference_records

ROOT = Path(__file__).resolve().parent
DEV = ROOT / "data/dev.csv"
REFERENCE = ROOT / "raw_data/dico_nli_dev_track1_reference.csv"
RESULTS = ROOT / "results.csv"


def load_saved_model(path):
    path = Path(path)
    settings_path = path / "training_config.json"
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    trust = settings.get("model") == "gte-en-mlm-base"
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=trust)
    model = AutoModelForSequenceClassification.from_pretrained(path, trust_remote_code=trust)
    return model, tokenizer, settings


def evaluate(model, tokenizer, *, max_length, batch_size):
    if not DEV.is_file():
        raise FileNotFoundError(f"Cleaned development data not found at {DEV}. Run python PreProcessing.py first.")
    if not REFERENCE.is_file():
        raise FileNotFoundError(f"Development reference not found at {REFERENCE}.")
    ids, pairs, labels = read_transform_dico(DEV)
    if labels is None or not len(ids) or len(set(ids)) != len(ids):
        raise ValueError("Development data must have labels and unique instance IDs")
    reference = validate_reference_records(read_reference_file(REFERENCE))
    ids_set = set(ids)
    missing = ids_set - set(reference)
    if missing:
        raise ValueError(f"Development IDs missing from reference: {sorted(missing)[:5]}")
    gold = {}
    for instance_id, label_id in zip(ids, labels):
        row = reference[instance_id]
        if row.label != LABELS[int(label_id)]:
            raise ValueError(f"Development label disagrees with reference for {instance_id}")
        gold[instance_id] = replace(row, reverse_pair_id=None) if row.reverse_pair_id not in ids_set else row
    gold = validate_reference_records(list(gold.values()))
    model_labels = model.config.id2label
    if {int(k): v for k, v in model_labels.items()} != dict(enumerate(LABELS)):
        raise ValueError("Model label mapping does not match DataHandler labels")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    predicted = []
    with torch.inference_mode():
        for start in range(0, len(ids), batch_size):
            batch = pairs[start:start + batch_size]
            inputs = tokenizer(batch[:, 0].tolist(), batch[:, 1].tolist(), padding=True,
                               truncation=True, max_length=max_length, return_tensors="pt")
            outputs = model(**{key: value.to(device) for key, value in inputs.items()})
            predicted.extend(outputs.logits.argmax(dim=-1).cpu().tolist())
    records = [PredictionRecord(instance_id=str(i), label=LABELS[int(label)]) for i, label in zip(ids, predicted)]
    predictions = validate_prediction_records(records, gold)
    return compute_evaluation(gold, predictions)


def print_report(report):
    payload = report.to_dict()
    payload["classification"].pop("macro_f1", None)
    payload["consistency"].pop("pair_errors", None)
    print(json.dumps(payload, indent=2))


# Columns added after the first results were recorded; older rows keep these blank.
LATER_COLUMNS = {"patience", "best_epoch", "deterministic", "group_id", "train_limit"}


def append_result(report, settings, loss_history, best_epoch, path=RESULTS, group_id=None):
    """Append one completed run. The CSV stores nested diagnostics as JSON.

    `group_id` links runs that differ only in seed; it defaults to the new run's own ID.
    """
    path = Path(path)
    fieldnames = ["id", "group_id", "time_utc", *settings.keys(), "weighted_f1", "soft_cons", "hard_cons",
                  "accuracy", "confusion_matrix", "pair_errors", "loss_history", "best_epoch"]
    next_id = 1
    if path.exists() and path.stat().st_size:
        with path.open(encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            old_fields = reader.fieldnames
            rows = list(reader)
        if old_fields != fieldnames:
            # Older runs predate some columns (see LATER_COLUMNS); leave those values blank.
            missing = set(fieldnames) - set(old_fields or [])
            if not old_fields or not set(old_fields) <= set(fieldnames) or not missing <= LATER_COLUMNS:
                raise ValueError(f"Existing results file has incompatible columns: {path}")
            with path.open("w", encoding="utf-8", newline="") as output:
                writer = csv.DictWriter(output, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
        next_id = max((int(row["id"]) for row in rows), default=0) + 1
    classification, consistency = report.classification, report.consistency
    row = {"id": next_id, "group_id": next_id if group_id is None else group_id, "time_utc": datetime.now(timezone.utc).isoformat(), **settings,
           "weighted_f1": report.weighted_f1, "soft_cons": report.soft_cons, "hard_cons": report.hard_cons,
           "accuracy": classification.accuracy, "best_epoch": best_epoch,
           "confusion_matrix": json.dumps(classification.confusion_matrix, sort_keys=True),
           "pair_errors": json.dumps(consistency.pair_errors), "loss_history": json.dumps(loss_history)}
    new_file = not path.exists() or not path.stat().st_size
    with path.open("a", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        if new_file:
            writer.writeheader()
        writer.writerow(row)
    return next_id
