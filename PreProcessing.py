"""
Clean the files in raw_data.

Run: python PreProcessing.py
Each split is cleaned independently. The official reference and submission
files are intentionally left untouched for use with the official scorer.
"""

import csv
from collections import Counter, defaultdict
from pathlib import Path

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


def clean_file(path, report):
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

    # Contradictions are resolved conservatively: remove every row for an
    # ambiguous unordered pair, rather than arbitrarily choosing a label.
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
            report.append((path.name, line, row["instance_id"], "inconsistent", inconsistent[line]))
            continue
        key = (row["text1"], row["text2"], row["label"])
        if key in seen:
            report.append((path.name, line, row["instance_id"], "duplicate", f"same ordered texts and label as CSV line {seen[key]}"))
            continue
        seen[key] = line
        kept.append(row)

    with (OUTPUT_DIR / path.name).open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(kept)
    return len(rows), len(kept)


def main():
    paths = sorted(RAW_DIR.glob("*_track1_participant_labeled.csv"))
    if not paths:
        raise FileNotFoundError(f"No track 1 participant labeled CSV files found in {RAW_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report = []
    for path in paths:
        _, kept = clean_file(path, report)
        print(f"{path.name}: kept {kept} rows")
    with (OUTPUT_DIR / "removed_rows.csv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(("source_file", "csv_line", "instance_id", "reason", "details"))
        writer.writerows(report)
    print("Report:", dict(Counter(entry[3] for entry in report)))


if __name__ == "__main__":
    main()
