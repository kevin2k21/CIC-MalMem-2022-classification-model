"""Select, tune, train, and evaluate a malware-memory classifier.

This script consumes the leakage-safe feature splits produced by
``scripts/feature_engineering.py``. Model and hyperparameter selection use the
training and validation partitions only. The test partition is evaluated once,
after the selected estimator is refitted on train + validation data.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "reports" / "model_training"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "malware_detector.joblib"
SUMMARY_PATH = REPORT_DIR / "training_summary.json"
COMPARISON_PATH = REPORT_DIR / "model_comparison.csv"
IMPORTANCE_PATH = REPORT_DIR / "feature_importance.csv"
REPORT_PATH = REPORT_DIR / "model_selection_training_report.md"

TARGET = "Class"
RANDOM_STATE = 42
CV_FOLDS = 3
DECISION_THRESHOLD = 0.5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_split(name: str) -> tuple[pd.DataFrame, pd.Series, Path]:
    path = DATA_DIR / f"{name}_selected.csv.gz"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run scripts/feature_engineering.py first."
        )
    frame = pd.read_csv(path)
    if TARGET not in frame:
        raise ValueError(f"{path} does not contain the {TARGET!r} target.")
    target = frame.pop(TARGET)
    if set(target.unique()) != {0, 1}:
        raise ValueError(f"{path} must have a binary 0/1 {TARGET} target.")
    if frame.columns.duplicated().any():
        raise ValueError(f"{path} contains duplicate feature names.")
    return frame, target.astype("int8"), path


def model_searches() -> dict[str, tuple[Pipeline, dict[str, list[Any]]]]:
    """Return compact, reproducible searches spanning linear and tree models."""
    return {
        "Logistic Regression": (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    (
                        "classifier",
                        LogisticRegression(max_iter=2_000, random_state=RANDOM_STATE),
                    ),
                ]
            ),
            {
                "classifier__C": [0.1, 1.0, 10.0],
                "classifier__class_weight": [None, "balanced"],
            },
        ),
        "Random Forest": (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "classifier",
                        RandomForestClassifier(
                            n_estimators=250,
                            random_state=RANDOM_STATE,
                            n_jobs=1,
                        ),
                    ),
                ]
            ),
            {
                "classifier__max_features": ["sqrt", 0.75],
                "classifier__min_samples_leaf": [1, 2],
                "classifier__class_weight": [None, "balanced"],
            },
        ),
        "Extra Trees": (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "classifier",
                        ExtraTreesClassifier(
                            n_estimators=250,
                            random_state=RANDOM_STATE,
                            n_jobs=1,
                        ),
                    ),
                ]
            ),
            {
                "classifier__max_features": ["sqrt", 0.75],
                "classifier__min_samples_leaf": [1, 2],
                "classifier__class_weight": [None, "balanced"],
            },
        ),
        "Histogram Gradient Boosting": (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "classifier",
                        HistGradientBoostingClassifier(
                            max_iter=200,
                            early_stopping=True,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
            {
                "classifier__learning_rate": [0.05, 0.1],
                "classifier__max_leaf_nodes": [15, 31],
                "classifier__l2_regularization": [0.0, 1.0],
            },
        ),
    }


def probabilities(model: BaseEstimator, features: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(features))[:, 1]
    scores = np.asarray(model.decision_function(features), dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(scores, -500, 500)))


def evaluate(model: BaseEstimator, features: pd.DataFrame, target: pd.Series) -> dict[str, Any]:
    scores = probabilities(model, features)
    prediction = (scores >= DECISION_THRESHOLD).astype("int8")
    matrix = confusion_matrix(target, prediction, labels=[0, 1])
    true_negative, false_positive, false_negative, true_positive = matrix.ravel()
    specificity = true_negative / (true_negative + false_positive)
    return {
        "accuracy": float(accuracy_score(target, prediction)),
        "precision": float(precision_score(target, prediction, zero_division=0)),
        "recall": float(recall_score(target, prediction, zero_division=0)),
        "specificity": float(specificity),
        "f1": float(f1_score(target, prediction, zero_division=0)),
        "f2": float(fbeta_score(target, prediction, beta=2, zero_division=0)),
        "roc_auc": float(roc_auc_score(target, scores)),
        "average_precision": float(average_precision_score(target, scores)),
        "confusion_matrix": matrix.tolist(),
        "true_negative": int(true_negative),
        "false_positive": int(false_positive),
        "false_negative": int(false_negative),
        "true_positive": int(true_positive),
    }


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def feature_importance(model: Pipeline, feature_names: list[str]) -> pd.DataFrame:
    classifier = model.named_steps["classifier"]
    if hasattr(classifier, "feature_importances_"):
        values = np.asarray(classifier.feature_importances_)
        method = "impurity-based feature importance"
    elif hasattr(classifier, "coef_"):
        values = np.abs(np.asarray(classifier.coef_)[0])
        method = "absolute standardized coefficient"
    else:
        return pd.DataFrame(columns=["rank", "feature", "importance", "method"])
    result = pd.DataFrame({"feature": feature_names, "importance": values})
    result = result.sort_values(["importance", "feature"], ascending=[False, True]).reset_index(drop=True)
    result.insert(0, "rank", np.arange(1, len(result) + 1))
    result["method"] = method
    return result


def report_table(comparison: pd.DataFrame) -> str:
    columns = [
        "model",
        "cv_f2_mean",
        "cv_recall_mean",
        "validation_f2",
        "validation_recall",
        "validation_precision",
        "validation_roc_auc",
    ]
    lines = [
        "| Model | CV F2 | CV recall | Validation F2 | Validation recall | Validation precision | Validation ROC-AUC |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in comparison[columns].itertuples(index=False):
        lines.append(
            f"| {row.model} | {row.cv_f2_mean:.5f} | {row.cv_recall_mean:.5f} | "
            f"{row.validation_f2:.5f} | {row.validation_recall:.5f} | "
            f"{row.validation_precision:.5f} | {row.validation_roc_auc:.5f} |"
        )
    return "\n".join(lines)


def write_report(summary: dict[str, Any], comparison: pd.DataFrame, importance: pd.DataFrame) -> None:
    test = summary["test_metrics"]
    selected = summary["selected_model"]
    top_features = "\n".join(
        f"{int(row.rank)}. `{row.feature}` ({row.importance:.6f})"
        for row in importance.head(10).itertuples()
    )
    if not top_features:
        top_features = "The selected estimator does not expose native feature importance."
    report = f"""# Model Selection and Training Report

## Outcome

**{selected}** was selected after training-only stratified {CV_FOLDS}-fold cross-validation and validation-set comparison. It was then refitted on the combined train + validation data and evaluated once on the untouched test set.

- Test accuracy: **{test['accuracy']:.4%}**
- Test malware recall: **{test['recall']:.4%}**
- Test precision: **{test['precision']:.4%}**
- Test F2 score: **{test['f2']:.4%}**
- Test ROC-AUC: **{test['roc_auc']:.6f}**
- Test confusion matrix: `{test['confusion_matrix']}` (`[[TN, FP], [FN, TP]]`)

## Model-selection decision

F2 is the primary metric because a missed malware sample (false negative) is more costly than an additional investigation, while F2 still penalizes excessive false alarms. ROC-AUC, precision, runtime, interpretability, and scalability are retained as decision evidence. The final winner is the candidate with the highest validation F2; validation ROC-AUC and lower fit time break ties.

{report_table(comparison)}

The candidates cover an interpretable linear baseline, two robust nonlinear tree ensembles, and gradient boosting. Each candidate received a compact hyperparameter search rather than a single default fit.

## Final training

- Training rows used for model search: {summary['rows']['train']:,}
- Validation rows used for final selection: {summary['rows']['validation']:,}
- Rows used in final refit: {summary['rows']['final_refit']:,}
- Untouched test rows: {summary['rows']['test']:,}
- Selected predictors: {summary['feature_count']}
- Decision threshold: {DECISION_THRESHOLD}
- Best parameters: `{json.dumps(summary['best_parameters'], sort_keys=True)}`
- Serialized model bundle: `models/malware_detector.joblib`

The saved bundle contains the fitted pipeline, exact ordered feature list, threshold, class meaning, training metadata, and input hashes. Median imputation (and scaling where required) remains inside the pipeline, so preprocessing learned during training is applied consistently at inference.

## Most influential features

Importance method: {importance['method'].iloc[0] if len(importance) else 'not available'}.

{top_features}

## Leakage and generalization controls

- Hyperparameters were tuned only with folds inside the training partition.
- Candidate comparison used validation data, never test data.
- The test set was scored only after model choice and final refit.
- Every preprocessing transform is fitted inside its model pipeline.
- Random seeds and input SHA-256 hashes are recorded for reproducibility.

The source EDA found near-perfect single-feature separation. Therefore, even this strong held-out result may be optimistic if related captures or malware families occur in multiple random splits. A family-separated, time-separated, or independently collected external test set is required before deployment.

## Reproduce

From the project root:

```bash
.venv/bin/python scripts/model_selection_training.py
```
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def main() -> None:
    train_x, train_y, train_path = load_split("train")
    validation_x, validation_y, validation_path = load_split("validation")
    test_x, test_y, test_path = load_split("test")
    if list(train_x.columns) != list(validation_x.columns) or list(train_x.columns) != list(test_x.columns):
        raise ValueError("Train, validation, and test feature schemas do not match.")
    if not np.isfinite(train_x.to_numpy(dtype=float)).all():
        raise ValueError("Training predictors contain a non-finite value.")

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "f2": make_scorer(fbeta_score, beta=2, zero_division=0),
        "recall": make_scorer(recall_score, zero_division=0),
        "precision": make_scorer(precision_score, zero_division=0),
        "roc_auc": "roc_auc",
    }
    comparison_rows: list[dict[str, Any]] = []
    fitted: dict[str, GridSearchCV] = {}

    for name, (pipeline, parameter_grid) in model_searches().items():
        started = time.perf_counter()
        search = GridSearchCV(
            pipeline,
            parameter_grid,
            scoring=scoring,
            refit="f2",
            cv=cv,
            n_jobs=-1,
            return_train_score=False,
            error_score="raise",
        )
        search.fit(train_x, train_y)
        elapsed = time.perf_counter() - started
        validation_metrics = evaluate(search.best_estimator_, validation_x, validation_y)
        best_index = search.best_index_
        comparison_rows.append(
            {
                "model": name,
                "tested_configurations": len(search.cv_results_["params"]),
                "fit_seconds": elapsed,
                "cv_f2_mean": search.cv_results_["mean_test_f2"][best_index],
                "cv_f2_std": search.cv_results_["std_test_f2"][best_index],
                "cv_recall_mean": search.cv_results_["mean_test_recall"][best_index],
                "cv_precision_mean": search.cv_results_["mean_test_precision"][best_index],
                "cv_roc_auc_mean": search.cv_results_["mean_test_roc_auc"][best_index],
                **{f"validation_{key}": value for key, value in validation_metrics.items() if not isinstance(value, list)},
                "validation_confusion_matrix": json.dumps(validation_metrics["confusion_matrix"]),
                "best_parameters": json.dumps(json_ready(search.best_params_), sort_keys=True),
            }
        )
        fitted[name] = search
        print(
            f"{name}: validation F2={validation_metrics['f2']:.6f}, "
            f"recall={validation_metrics['recall']:.6f}, elapsed={elapsed:.1f}s",
            flush=True,
        )

    comparison = pd.DataFrame(comparison_rows).sort_values(
        ["validation_f2", "validation_roc_auc", "fit_seconds"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    comparison.insert(0, "rank", np.arange(1, len(comparison) + 1))
    selected_name = str(comparison.loc[0, "model"])
    selected_search = fitted[selected_name]

    final_x = pd.concat([train_x, validation_x], ignore_index=True)
    final_y = pd.concat([train_y, validation_y], ignore_index=True)
    final_model = selected_search.best_estimator_.set_params(**selected_search.best_params_)
    final_started = time.perf_counter()
    final_model.fit(final_x, final_y)
    final_fit_seconds = time.perf_counter() - final_started
    test_metrics = evaluate(final_model, test_x, test_y)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    importance = feature_importance(final_model, list(final_x.columns))
    comparison.to_csv(COMPARISON_PATH, index=False)
    importance.to_csv(IMPORTANCE_PATH, index=False)

    summary = {
        "selected_model": selected_name,
        "selection_metric": "validation_f2",
        "selection_rationale": "F2 emphasizes malware recall while retaining a precision penalty.",
        "best_parameters": json_ready(selected_search.best_params_),
        "random_state": RANDOM_STATE,
        "cross_validation_folds": CV_FOLDS,
        "decision_threshold": DECISION_THRESHOLD,
        "feature_count": final_x.shape[1],
        "feature_names": list(final_x.columns),
        "rows": {
            "train": len(train_x),
            "validation": len(validation_x),
            "final_refit": len(final_x),
            "test": len(test_x),
        },
        "class_counts": {
            "final_refit": {str(key): int(value) for key, value in final_y.value_counts().sort_index().items()},
            "test": {str(key): int(value) for key, value in test_y.value_counts().sort_index().items()},
        },
        "input_files": {
            str(path.relative_to(PROJECT_ROOT)): sha256(path)
            for path in [train_path, validation_path, test_path]
        },
        "candidate_results": json_ready(comparison.to_dict(orient="records")),
        "final_fit_seconds": final_fit_seconds,
        "test_metrics": test_metrics,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
            "platform": platform.platform(),
        },
    }
    bundle = {
        "model": final_model,
        "feature_names": list(final_x.columns),
        "target": TARGET,
        "decision_threshold": DECISION_THRESHOLD,
        "class_labels": {0: "Benign", 1: "Malware"},
        "metadata": {
            "selected_model": selected_name,
            "best_parameters": json_ready(selected_search.best_params_),
            "random_state": RANDOM_STATE,
            "input_sha256": summary["input_files"],
        },
    }
    joblib.dump(bundle, MODEL_PATH, compress=3)
    summary["model_file"] = str(MODEL_PATH.relative_to(PROJECT_ROOT))
    summary["model_sha256"] = sha256(MODEL_PATH)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_report(summary, comparison, importance)
    print(
        f"Selected {selected_name}; test F2={test_metrics['f2']:.6f}, "
        f"recall={test_metrics['recall']:.6f}; saved {MODEL_PATH.relative_to(PROJECT_ROOT)}",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Training failed: {error}", file=sys.stderr)
        raise
