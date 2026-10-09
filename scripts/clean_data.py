"""Clean the CIC-MalMem-2022 working copy without modifying the raw dataset."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "Obfuscated-MalMem2022.csv"
WORKING_PATH = PROJECT_ROOT / "data" / "cleansed_data.csv"
AUDIT_PATH = PROJECT_ROOT / "reports" / "data_cleaning_audit.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def profile(frame: pd.DataFrame) -> dict[str, object]:
    numeric = frame.select_dtypes(include=np.number)
    infinite_cells = int(np.isinf(numeric.to_numpy()).sum())
    return {
        "rows": int(frame.shape[0]),
        "columns": int(frame.shape[1]),
        "missing_cells": int(frame.isna().sum().sum()),
        "infinite_numeric_cells": infinite_cells,
        "exact_duplicate_rows": int(frame.duplicated().sum()),
        "class_counts": {
            str(label): int(count)
            for label, count in frame["Class"].value_counts(dropna=False).items()
        },
    }


def main() -> None:
    if not RAW_PATH.exists() or not WORKING_PATH.exists():
        raise FileNotFoundError("Both the raw dataset and working copy are required.")

    raw_hash_before = sha256(RAW_PATH)
    working_hash_before = sha256(WORKING_PATH)
    # Always regenerate the working file from the immutable source. This keeps
    # the procedure idempotent and prevents repeated cleaning from drifting.
    frame = pd.read_csv(RAW_PATH)
    before = profile(frame)

    # Normalize labels and column names without changing numerical measurements.
    frame.columns = frame.columns.str.strip()
    for column in frame.select_dtypes(include="object").columns:
        frame[column] = frame[column].str.strip()

    # Convert non-finite numeric values to missing values so they cannot pass silently.
    frame.replace([np.inf, -np.inf], np.nan, inplace=True)

    # The benchmark currently contains no missing values. Fail explicitly if that
    # changes so imputation can be fitted on training data rather than leaked here.
    missing_cells = int(frame.isna().sum().sum())
    if missing_cells:
        raise ValueError(
            f"Found {missing_cells} missing cells. Handle them in the training pipeline."
        )

    feature_columns = [c for c in frame.columns if c not in {"Category", "Class"}]
    constant_columns = [
        column for column in feature_columns if frame[column].nunique(dropna=False) <= 1
    ]
    frame.drop(columns=constant_columns, inplace=True)

    # Category contains malware family/sample information and would reveal the target.
    frame.drop(columns=["Category"], inplace=True)

    label_mapping = {"Benign": 0, "Malware": 1}
    frame["Class"] = frame["Class"].map(label_mapping)
    if frame["Class"].isna().any():
        raise ValueError("Class contains an unexpected label.")
    frame["Class"] = frame["Class"].astype("int8")

    # Remove duplicates after excluding identifiers and constant fields. This
    # catches records that are distinct only because of the removed Category.
    rows_before_deduplication = len(frame)
    frame.drop_duplicates(inplace=True)
    frame.reset_index(drop=True, inplace=True)
    duplicates_removed = rows_before_deduplication - len(frame)

    temp_path = WORKING_PATH.with_suffix(".tmp.csv")
    frame.to_csv(temp_path, index=False)
    temp_path.replace(WORKING_PATH)

    raw_hash_after = sha256(RAW_PATH)
    if raw_hash_before != raw_hash_after:
        raise RuntimeError("The raw dataset changed during cleaning.")

    after = profile(frame)
    audit = {
        "source_file": str(RAW_PATH.relative_to(PROJECT_ROOT)),
        "cleaned_file": str(WORKING_PATH.relative_to(PROJECT_ROOT)),
        "raw_sha256_before": raw_hash_before,
        "raw_sha256_after": raw_hash_after,
        "working_sha256_before": working_hash_before,
        "working_sha256_after": sha256(WORKING_PATH),
        "before": before,
        "after": after,
        "operations": {
            "trimmed_column_names_and_text": True,
            "infinite_values_converted_to_missing": True,
            "missing_values_imputed": 0,
            "duplicates_removed": int(duplicates_removed),
            "constant_columns_removed": constant_columns,
            "leakage_columns_removed": ["Category"],
            "label_encoding": label_mapping,
            "scaling_deferred_until_after_train_test_split": True,
        },
    }

    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_PATH.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
