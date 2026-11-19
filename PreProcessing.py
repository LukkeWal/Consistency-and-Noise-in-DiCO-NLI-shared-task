"""
Clean the files in raw_data.

Run: python PreProcessing.py
Each split is cleaned independently. The official reference and submission
files are intentionally left untouched for use with the official scorer.
"""

import csv
from collections import Counter, defaultdict
from pathlib import Path
import random

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "raw_data"
OUTPUT_DIR = ROOT / "data"
REQUIRED_COLUMNS = ("instance_id", "pair_id", "text1_lang", "text2_lang", "text1", "text2", "label")
REVERSE_LABEL = {
    "EQUIVALENCE": "EQUIVALENCE",
    "FORWARD_ENTAILMENT": "BACKWARD_ENTAILMENT",
    "BACKWARD_ENTAILMENT": "FORWARD_ENTAILMENT",
}
VALID_LABELS = set(REVERSE_LABEL) | {"NEGATIVE_OTHER"}
VALIDATION_SIZE = 660
SPLIT_SEED = 42


def clean_file(path, report):
    is_dev = "_dev_track1_" in path.name
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)) or not set(REQUIRED_COLUMNS).issubset(fields):
            raise ValueError(f"{path.name}: expected columns {REQUIRED_COLUMNS}; got {fields}")
        rows = []
        for row in reader:
            line = reader.line_num
            if None in row:
                raise ValueError(f"{path.name}: too many fields at CSV line {line}")
            missing = [field for field in fields if row[field] is None or not row[field].strip()]
            if missing:
                report.append((path.name, line, row.get("instance_id", ""), "missing values", ", ".join(missing)))
                continue
            label = row["label"]
            if label not in VALID_LABELS:
                raise ValueError(f"{path.name}: invalid label {label!r} at CSV line {line}")
            rows.append((line, row))

    # Identify every row belonging to an ambiguous unordered pair.
    groups = defaultdict(list)
    for line, row in rows:
        groups[tuple(sorted((row["text1"], row["text2"])))].append((line, row))

    inconsistent = {}
    for group in groups.values():
        labels_by_order = defaultdict(set)
        for _, row in group:
            labels_by_order[(row["text1"], row["text2"])].add(row["label"])
        conflict = any(len(labels) > 1 for labels in labels_by_order.values())
        for (first, second), labels in labels_by_order.items():
            reverse_labels = labels_by_order.get((second, first), set())
            # NEGATIVE_OTHER has no defined deterministic reverse in the task.
            for label in labels:
                if label in REVERSE_LABEL and any(
                    other in REVERSE_LABEL and other != REVERSE_LABEL[label]
                    for other in reverse_labels
                ):
                    conflict = True
        if conflict:
            conflicts = []
            for index, (line, row) in enumerate(group):
                for other_line, other in group[index + 1:]:
                    same_order = (row["text1"], row["text2"]) == (other["text1"], other["text2"])
                    if same_order and row["label"] != other["label"]:
                        conflicts.append((line, other_line))
                    elif not same_order and row["label"] in REVERSE_LABEL and other["label"] in REVERSE_LABEL and other["label"] != REVERSE_LABEL[row["label"]]:
                        conflicts.append((line, other_line))
                if row["text1"] == row["text2"] and row["label"] in REVERSE_LABEL and row["label"] != REVERSE_LABEL[row["label"]]:
                    conflicts.append((line, line))
            details = "; ".join(f"CSV line {a} conflicts with CSV line {b}" for a, b in conflicts)
            for line, _ in group:
                inconsistent[line] = details

    kept = []
    seen = {}
    for line, row in rows:
        if line in inconsistent:
            if is_dev:
                print(f"WARNING: {path.name}, CSV line {line}, {row['instance_id']}: inconsistent: {inconsistent[line]} (kept)")
            else:
                report.append((path.name, line, row["instance_id"], "inconsistent", inconsistent[line]))
                continue
        key = (row["text1"], row["text2"], row["label"])
        if key in seen:
            details = f"same ordered texts and label as CSV line {seen[key]}"
            if is_dev:
                print(f"WARNING: {path.name}, CSV line {line}, {row['instance_id']}: duplicate: {details} (kept)")
            else:
                report.append((path.name, line, row["instance_id"], "duplicate", details))
                continue
        else:
            seen[key] = line
        kept.append(row)

    output_path = OUTPUT_DIR / ("dev.csv" if is_dev else "train.csv")
    with output_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(kept)
    return len(rows), len(kept)


def split_training_data(path):
    """Create an exact-size, label-stratified split without separating pairs."""
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        fields = reader.fieldnames
        rows = list(reader)
    if len(rows) <= VALIDATION_SIZE:
        raise ValueError(f"Need more than {VALIDATION_SIZE} clean training rows to split")

    # Join rows sharing a source pair or the same unordered texts, including
    # transitive links, so neither source pairs nor reversals cross the split.
    parents = list(range(len(rows)))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    seen = {}
    for i, row in enumerate(rows):
        for key in (("pair", row["pair_id"]), ("texts", *sorted((row["text1"], row["text2"])))):
            if key in seen:
                parents[find(i)] = find(seen[key])
            else:
                seen[key] = i
    groups = defaultdict(list)
    for i in range(len(rows)):
        groups[find(i)].append(i)

    labels = sorted(VALID_LABELS)
    patterns = defaultdict(list)
    for indices in groups.values():
        counts = Counter(rows[i]["label"] for i in indices)
        patterns[tuple(counts[label] for label in labels)].append(indices)
    patterns = dict(sorted(patterns.items()))
    vectors = np.array(list(patterns), dtype=float)
    n = len(vectors)
    totals = np.array([sum(row["label"] == label for row in rows) for label in labels])
    targets = totals * (VALIDATION_SIZE / len(rows))
    # Integer variables choose how many groups of each pattern to include;
    # continuous variables represent absolute deviation from label targets.
    objective = np.r_[np.zeros(n), np.ones(len(labels))]
    sizes = vectors.sum(axis=1)
    constraints = [LinearConstraint(np.r_[sizes, np.zeros(len(labels))], VALIDATION_SIZE, VALIDATION_SIZE)]
    for j, target in enumerate(targets):
        slack = np.zeros(len(labels))
        slack[j] = -1
        constraints.append(LinearConstraint(np.r_[vectors[:, j], slack], -np.inf, target))
        constraints.append(LinearConstraint(np.r_[-vectors[:, j], slack], -np.inf, -target))
    result = milp(objective, integrality=np.r_[np.ones(n), np.zeros(len(labels))],
                  bounds=Bounds(np.zeros(n + len(labels)),
                                np.r_[[len(items) for items in patterns.values()], np.full(len(labels), np.inf)]),
                  constraints=constraints)
    if not result.success:
        raise ValueError(f"Cannot create a {VALIDATION_SIZE}-row grouped validation split: {result.message}")
    rng = random.Random(SPLIT_SEED)
    val_indices = set()
    for items, count in zip(patterns.values(), np.rint(result.x[:n]).astype(int)):
        rng.shuffle(items)
        for group in items[:count]:
            val_indices.update(group)
    if len(val_indices) != VALIDATION_SIZE:
        raise RuntimeError("Validation split has an unexpected number of rows")
    val_path = OUTPUT_DIR / "val.csv"
    for output_path, select_val in ((path, False), (val_path, True)):
        with output_path.open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            writer.writerows(row for i, row in enumerate(rows) if (i in val_indices) == select_val)
    print(f"Training split: {len(rows) - len(val_indices)} rows; validation split: {len(val_indices)} rows (seed {SPLIT_SEED})")
    print("Validation labels:", dict(Counter(rows[i]["label"] for i in val_indices)))


def main():
    paths = sorted(RAW_DIR.glob("*_track1_participant_labeled.csv"))
    if not paths:
        raise FileNotFoundError(f"No track 1 participant labeled CSV files found in {RAW_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report = []
    for path in paths:
        _, kept = clean_file(path, report)
        print(f"{path.name}: kept {kept} rows")
        if "_train_track1_" in path.name:
            split_training_data(OUTPUT_DIR / "train.csv")
    with (OUTPUT_DIR / "removed_rows.csv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(("source_file", "csv_line", "instance_id", "reason", "details"))
        writer.writerows(report)
    print("Report:", dict(Counter(entry[3] for entry in report)))


if __name__ == "__main__":
    main()
