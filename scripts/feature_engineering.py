"""Engineer and select CIC-MalMem-2022 features without target leakage.

The input is split before any target-aware operation. Domain features are
deterministic row-level ratios, while mutual-information scoring and
correlation pruning are fitted on the training partition only.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "cleansed_data.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "feature_engineering"
REPORT_PATH = OUTPUT_DIR / "feature_selection_report.md"
SUMMARY_PATH = OUTPUT_DIR / "feature_selection_summary.json"
SCORES_PATH = OUTPUT_DIR / "feature_scores.csv"
PRUNING_PATH = OUTPUT_DIR / "correlation_pruning.csv"

TARGET = "Class"
RANDOM_STATE = 42
TEST_SIZE = 0.20
VALIDATION_SHARE_OF_REMAINDER = 0.20
CORRELATION_THRESHOLD = 0.95
TOP_K = 25


FEATURE_DEFINITIONS = {
    "derived.handles_per_process": "handles.nhandles / pslist.nproc",
    "derived.threads_per_process": "handles.nthread / pslist.nproc",
    "derived.events_per_process": "handles.nevent / pslist.nproc",
    "derived.mutants_per_process": "handles.nmutant / pslist.nproc",
    "derived.sections_per_process": "handles.nsection / pslist.nproc",
    "derived.dlls_per_process_recomputed": "dlllist.ndlls / pslist.nproc",
    "derived.unloaded_module_rate": "ldrmodules.not_in_load / dlllist.ndlls",
    "derived.uninitialized_module_rate": "ldrmodules.not_in_init / dlllist.ndlls",
    "derived.unmapped_module_rate": "ldrmodules.not_in_mem / dlllist.ndlls",
    "derived.injections_per_process": "malfind.ninjections / pslist.nproc",
    "derived.unique_injection_rate": "malfind.uniqueInjections / malfind.ninjections",
    "derived.commit_charge_per_injection": "malfind.commitCharge / malfind.ninjections",
    "derived.psxview_discrepancies_per_process": "sum of seven psxview discrepancy counts / pslist.nproc",
    "derived.active_service_rate": "svcscan.nactive / svcscan.nservices",
    "derived.driver_service_rate": "(svcscan.kernel_drivers + svcscan.fs_drivers) / svcscan.nservices",
    "derived.process_service_rate": "svcscan.process_services / svcscan.nservices",
    "derived.shared_process_service_rate": "svcscan.shared_process_services / svcscan.nservices",
    "derived.anonymous_callback_rate": "callbacks.nanonymous / callbacks.ncallbacks",
    "derived.generic_callback_rate": "callbacks.ngeneric / callbacks.ncallbacks",
    "derived.file_handle_rate": "handles.nfile / handles.nhandles",
    "derived.event_handle_rate": "handles.nevent / handles.nhandles",
    "derived.thread_handle_rate": "handles.nthread / handles.nhandles",
    "derived.key_handle_rate": "handles.nkey / handles.nhandles",
    "derived.mutant_handle_rate": "handles.nmutant / handles.nhandles",
    "derived.section_handle_rate": "handles.nsection / handles.nhandles",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide two aligned series, returning zero when the denominator is zero."""
    result = np.divide(
        numerator.to_numpy(dtype=float),
        denominator.to_numpy(dtype=float),
        out=np.zeros(len(numerator), dtype=float),
        where=denominator.to_numpy(dtype=float) != 0,
    )
    return pd.Series(result, index=numerator.index)


def engineer_features(features: pd.DataFrame) -> pd.DataFrame:
    """Add domain-derived, row-level features without using the target."""
    result = features.copy()
    nproc = result["pslist.nproc"]
    nhandles = result["handles.nhandles"]
    ndlls = result["dlllist.ndlls"]
    nservices = result["svcscan.nservices"]
    ncallbacks = result["callbacks.ncallbacks"]
    ninjections = result["malfind.ninjections"]

    result["derived.handles_per_process"] = safe_divide(nhandles, nproc)
    result["derived.threads_per_process"] = safe_divide(result["handles.nthread"], nproc)
    result["derived.events_per_process"] = safe_divide(result["handles.nevent"], nproc)
    result["derived.mutants_per_process"] = safe_divide(result["handles.nmutant"], nproc)
    result["derived.sections_per_process"] = safe_divide(result["handles.nsection"], nproc)
    result["derived.dlls_per_process_recomputed"] = safe_divide(ndlls, nproc)
    result["derived.unloaded_module_rate"] = safe_divide(result["ldrmodules.not_in_load"], ndlls)
    result["derived.uninitialized_module_rate"] = safe_divide(result["ldrmodules.not_in_init"], ndlls)
    result["derived.unmapped_module_rate"] = safe_divide(result["ldrmodules.not_in_mem"], ndlls)
    result["derived.injections_per_process"] = safe_divide(ninjections, nproc)
    result["derived.unique_injection_rate"] = safe_divide(
        result["malfind.uniqueInjections"], ninjections
    )
    result["derived.commit_charge_per_injection"] = safe_divide(
        result["malfind.commitCharge"], ninjections
    )

    psxview_counts = [
        "psxview.not_in_pslist",
        "psxview.not_in_eprocess_pool",
        "psxview.not_in_ethread_pool",
        "psxview.not_in_pspcid_list",
        "psxview.not_in_csrss_handles",
        "psxview.not_in_session",
        "psxview.not_in_deskthrd",
    ]
    result["derived.psxview_discrepancies_per_process"] = safe_divide(
        result[psxview_counts].sum(axis=1), nproc
    )

    result["derived.active_service_rate"] = safe_divide(result["svcscan.nactive"], nservices)
    result["derived.driver_service_rate"] = safe_divide(
        result["svcscan.kernel_drivers"] + result["svcscan.fs_drivers"], nservices
    )
    result["derived.process_service_rate"] = safe_divide(
        result["svcscan.process_services"], nservices
    )
    result["derived.shared_process_service_rate"] = safe_divide(
        result["svcscan.shared_process_services"], nservices
    )
    result["derived.anonymous_callback_rate"] = safe_divide(
        result["callbacks.nanonymous"], ncallbacks
    )
    result["derived.generic_callback_rate"] = safe_divide(
        result["callbacks.ngeneric"], ncallbacks
    )

    for short_name, source in {
        "file": "handles.nfile",
        "event": "handles.nevent",
        "thread": "handles.nthread",
        "key": "handles.nkey",
        "mutant": "handles.nmutant",
        "section": "handles.nsection",
    }.items():
        result[f"derived.{short_name}_handle_rate"] = safe_divide(
            result[source], nhandles
        )

    values = result.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Feature engineering produced a non-finite value.")
    return result


def select_features(
    train_features: pd.DataFrame, train_target: pd.Series
) -> tuple[list[str], pd.DataFrame, pd.DataFrame]:
    """Rank by train-only mutual information, then remove correlated duplicates."""
    scores = mutual_info_classif(
        train_features,
        train_target,
        discrete_features=False,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    score_table = pd.DataFrame(
        {
            "feature": train_features.columns,
            "mutual_information": scores,
            "is_engineered": train_features.columns.str.startswith("derived."),
        }
    ).sort_values(["mutual_information", "feature"], ascending=[False, True])

    correlation = train_features.corr().abs()
    retained: list[str] = []
    pruning_rows: list[dict[str, object]] = []
    for feature in score_table["feature"]:
        if not retained:
            retained.append(feature)
            continue
        correlations = correlation.loc[feature, retained]
        partner = correlations.idxmax()
        maximum = float(correlations.max())
        if maximum >= CORRELATION_THRESHOLD:
            pruning_rows.append(
                {
                    "dropped_feature": feature,
                    "retained_feature": partner,
                    "absolute_train_correlation": maximum,
                    "reason": f"absolute correlation >= {CORRELATION_THRESHOLD:.2f}",
                }
            )
        else:
            retained.append(feature)

    selected = retained[: min(TOP_K, len(retained))]
    score_table["survived_correlation_pruning"] = score_table["feature"].isin(retained)
    score_table["selected"] = score_table["feature"].isin(selected)
    score_table["selection_rank"] = pd.Series(
        {feature: rank for rank, feature in enumerate(selected, start=1)}
    ).reindex(score_table["feature"]).to_numpy()
    pruning_table = pd.DataFrame(pruning_rows)
    return selected, score_table, pruning_table


def build_model() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(max_iter=2_000, random_state=RANDOM_STATE),
            ),
        ]
    )


def evaluate(model: Pipeline, features: pd.DataFrame, target: pd.Series) -> dict[str, object]:
    probabilities = model.predict_proba(features)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    matrix = confusion_matrix(target, predictions, labels=[0, 1])
    return {
        "accuracy": float(accuracy_score(target, predictions)),
        "precision": float(precision_score(target, predictions, zero_division=0)),
        "recall": float(recall_score(target, predictions, zero_division=0)),
        "f1": float(f1_score(target, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(target, probabilities)),
        "confusion_matrix": matrix.tolist(),
    }


def metrics_table(results: dict[str, dict[str, dict[str, object]]]) -> str:
    lines = [
        "| Feature set | Split | Accuracy | Precision | Recall | F1 | ROC-AUC |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for feature_set, split_results in results.items():
        for split_name, values in split_results.items():
            lines.append(
                f"| {feature_set} | {split_name} | {values['accuracy']:.4f} | "
                f"{values['precision']:.4f} | {values['recall']:.4f} | "
                f"{values['f1']:.4f} | {values['roc_auc']:.5f} |"
            )
    return "\n".join(lines)


def write_report(summary: dict[str, object], score_table: pd.DataFrame) -> None:
    selected_rows = score_table.loc[score_table["selected"]].sort_values("selection_rank")
    feature_lines = "\n".join(
        f"{int(row.selection_rank)}. `{row.feature}` "
        f"(MI={row.mutual_information:.4f}, {'engineered' if row.is_engineered else 'original'})"
        for row in selected_rows.itertuples()
    )
    engineered_selected = int(selected_rows["is_engineered"].sum())
    report = f"""# Feature Engineering and Selection Report

## Outcome

The process added **{summary['engineered_feature_count']} domain features** to the 52 original measurements, then selected **{summary['selected_feature_count']} features** using only the training partition. Of those, **{engineered_selected} are engineered features**. Correlation pruning removed **{summary['correlation_pruned_count']} redundant candidates** at an absolute training correlation threshold of {CORRELATION_THRESHOLD:.2f}.

The exported train, validation, and test files contain only the selected predictors and `Class`. Their separation must be preserved during modeling.

## Leakage controls

- A stratified 64%/16%/20% train/validation/test split was created before target-aware selection.
- Ratios are deterministic row-level transformations and do not use `Class` or dataset-wide statistics.
- Mutual information and correlation pruning were fitted on the training partition only.
- Scaling and median-imputation safeguards were fitted inside each training-only logistic-regression pipeline.
- The raw and cleansed source files were not modified.

## Selected features

{feature_lines}

## Held-out benchmark

This comparison uses the same split and the same regularized logistic-regression estimator. The baseline uses all 52 original features; the selected model uses the engineered-and-pruned set.

{metrics_table(summary['benchmark'])}

Confusion matrices use `[[true benign, false malware], [missed malware, detected malware]]` ordering.

## Feature engineering rationale

- Per-process rates normalize counts for differing process populations.
- Module discrepancy rates represent hidden/unlinked modules relative to the DLL population.
- Injection features capture injection density, uniqueness, and memory commitment.
- PSXView discrepancy density summarizes disagreement across process-enumeration views.
- Service and callback rates describe composition rather than raw machine size.
- Handle subtype rates represent the process's object-use profile.

Outliers were retained because the EDA identified them as potentially meaningful attack signal. No target-derived encoding, family/category field, PCA component, or full-dataset scaling was introduced.

## Important limitation

The EDA found near-perfect single-feature separation. These random-row holdout scores may still be optimistic if related captures or malware families occur across splits. Before deployment, repeat evaluation on a time-separated, family-separated, or independently collected dataset.

## Reproducibility

- Input: `data/cleansed_data.csv`
- Input SHA-256: `{summary['source_sha256']}`
- Random seed: {RANDOM_STATE}
- Script: `scripts/feature_engineering.py`
- Selected split files: `data/processed/*_selected.csv.gz`

Run from the project root:

```bash
.venv/bin/python scripts/feature_engineering.py
```
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing input dataset: {DATA_PATH}")

    frame = pd.read_csv(DATA_PATH)
    if TARGET not in frame:
        raise ValueError(f"Input dataset must contain {TARGET!r}.")
    if set(frame[TARGET].unique()) != {0, 1}:
        raise ValueError("Class must be binary and encoded as 0/1.")

    original_features = frame.drop(columns=TARGET)
    target = frame[TARGET].astype("int8")
    train_x, test_x, train_y, test_y = train_test_split(
        original_features,
        target,
        test_size=TEST_SIZE,
        stratify=target,
        random_state=RANDOM_STATE,
    )
    train_x, validation_x, train_y, validation_y = train_test_split(
        train_x,
        train_y,
        test_size=VALIDATION_SHARE_OF_REMAINDER,
        stratify=train_y,
        random_state=RANDOM_STATE,
    )

    engineered_train = engineer_features(train_x)
    engineered_validation = engineer_features(validation_x)
    engineered_test = engineer_features(test_x)
    selected, scores, pruning = select_features(engineered_train, train_y)

    baseline_model = build_model().fit(train_x, train_y)
    selected_model = build_model().fit(engineered_train[selected], train_y)
    benchmark = {
        "Original 52": {
            "validation": evaluate(baseline_model, validation_x, validation_y),
            "test": evaluate(baseline_model, test_x, test_y),
        },
        f"Selected {len(selected)}": {
            "validation": evaluate(
                selected_model, engineered_validation[selected], validation_y
            ),
            "test": evaluate(selected_model, engineered_test[selected], test_y),
        },
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    scores.to_csv(SCORES_PATH, index=False)
    pruning.to_csv(PRUNING_PATH, index=False)

    split_frames = {
        "train": (engineered_train[selected], train_y),
        "validation": (engineered_validation[selected], validation_y),
        "test": (engineered_test[selected], test_y),
    }
    for split_name, (split_features, split_target) in split_frames.items():
        output = split_features.copy()
        output[TARGET] = split_target
        output.to_csv(
            PROCESSED_DIR / f"{split_name}_selected.csv.gz",
            index=False,
            compression={"method": "gzip", "mtime": 0},
        )

    summary = {
        "source_file": str(DATA_PATH.relative_to(PROJECT_ROOT)),
        "source_sha256": sha256(DATA_PATH),
        "random_state": RANDOM_STATE,
        "split_rows": {
            "train": len(train_x),
            "validation": len(validation_x),
            "test": len(test_x),
        },
        "split_class_counts": {
            name: {str(key): int(value) for key, value in split_target.value_counts().sort_index().items()}
            for name, (_, split_target) in split_frames.items()
        },
        "original_feature_count": original_features.shape[1],
        "engineered_feature_count": len(FEATURE_DEFINITIONS),
        "candidate_feature_count": engineered_train.shape[1],
        "correlation_threshold": CORRELATION_THRESHOLD,
        "correlation_pruned_count": len(pruning),
        "top_k": TOP_K,
        "selected_feature_count": len(selected),
        "selected_features": selected,
        "selected_engineered_features": [
            feature for feature in selected if feature.startswith("derived.")
        ],
        "feature_definitions": FEATURE_DEFINITIONS,
        "benchmark": benchmark,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_report(summary, scores)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
