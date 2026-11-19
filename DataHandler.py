"""Load DiCo-NLI text pairs and encode labels for a classifier.

The return format and label order follow the SemEval starter kit. Tokenization
is left to the model's tokenizer so that the two texts remain separate inputs.
"""

import csv
from pathlib import Path

import numpy as np

# This is the alphabetical order used by the starter kit's LabelEncoder.
LABELS = ("BACKWARD_ENTAILMENT", "EQUIVALENCE", "FORWARD_ENTAILMENT", "NEGATIVE_OTHER")
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}


def read_transform_dico(path):
    """Return (instance IDs, ordered text pairs, encoded labels or None).

    `path` may point to a labeled clean CSV or an unlabeled inference CSV.
    Returned NumPy arrays preserve file order, including reversed text pairs.
    """
    path = Path(path)
    ids, pairs, labels = [], [], []
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        fields = reader.fieldnames or []
        required = {"instance_id", "text1", "text2"}
        if len(fields) != len(set(fields)) or not required.issubset(fields):
            raise ValueError(f"{path}: expected unique columns {sorted(required)}; got {fields}")
        has_labels = "label" in fields
        for row in reader:
            if None in row:
                raise ValueError(f"{path}: too many fields at CSV line {reader.line_num}")
            values = [" ".join((row.get(field) or "").split()) for field in ("instance_id", "text1", "text2")]
            if not all(values):
                raise ValueError(f"{path}: missing instance ID or text at CSV line {reader.line_num}")
            ids.append(values[0])
            pairs.append(values[1:])
            if has_labels:
                label = " ".join((row.get("label") or "").split())
                if label not in LABEL_TO_ID:
                    raise ValueError(f"{path}: invalid label {label!r} at CSV line {reader.line_num}")
                labels.append(LABEL_TO_ID[label])
    return (np.asarray(ids, dtype=str), np.asarray(pairs, dtype=str).reshape(-1, 2),
            np.asarray(labels, dtype=np.int64) if has_labels else None)


def decode_labels(label_ids):
    """Convert predicted integer class IDs back to official label strings."""
    decoded = []
    for label_id in label_ids:
        if not isinstance(label_id, (int, np.integer)) or isinstance(label_id, (bool, np.bool_)) or not 0 <= label_id < len(LABELS):
            raise ValueError(f"Invalid label ID: {label_id!r}")
        decoded.append(LABELS[label_id])
    return decoded
