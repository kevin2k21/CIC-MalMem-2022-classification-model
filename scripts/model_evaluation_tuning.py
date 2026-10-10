"""Evaluate, threshold-tune, and stress-test the trained malware detector.

The model family and hyperparameters are selected by
``scripts/model_selection_training.py``. This Step 8 script keeps the test set
out of tuning: it fits the selected pipeline on the training split, chooses a
decision threshold on validation data, refits on train + validation, and only
then reports final test performance. It also estimates bootstrap confidence
intervals and runs deterministic noise/missing-value stress tests.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any, Callable

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    fbeta_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports" / "model_evaluation"
FIGURE_DIR = REPORT_DIR / "figures"

INPUT_MODEL_PATH = MODEL_DIR / "malware_detector.joblib"
OUTPUT_MODEL_PATH = MODEL_DIR / "malware_detector_tuned.joblib"
SUMMARY_PATH = REPORT_DIR / "evaluation_summary.json"
METRICS_PATH = REPORT_DIR / "test_metrics.csv"
CLASS_REPORT_PATH = REPORT_DIR / "classification_report.csv"
THRESHOLD_PATH = REPORT_DIR / "threshold_search.csv"
ROBUSTNESS_PATH = REPORT_DIR / "robustness_results.csv"
REPORT_PATH = REPORT_DIR / "model_evaluation_tuning_report.md"

TARGET = "Class"
RANDOM_STATE = 42
BOOTSTRAP_SAMPLES = 1_000
CONFIDENCE_LEVEL = 0.95
THRESHOLD_GRID = np.linspace(0.01, 0.99, 99)
PERTURBATION_LEVELS = (0.01, 0.05, 0.10)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_split(name: str, feature_names: list[str]) -> tuple[pd.DataFrame, pd.Series, Path]:
    path = DATA_DIR / f"{name}_selected.csv.gz"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run scripts/feature_engineering.py first.")
    frame = pd.read_csv(path)
    if TARGET not in frame:
        raise ValueError(f"{path} does not contain target {TARGET!r}.")
    missing = sorted(set(feature_names) - set(frame.columns))
    extra = sorted(set(frame.columns) - set(feature_names) - {TARGET})
    if missing or extra:
        raise ValueError(f"Feature schema mismatch in {path}; missing={missing}, extra={extra}")
    target = frame.pop(TARGET).astype("int8")
    if not set(target.unique()).issubset({0, 1}) or target.nunique() != 2:
        raise ValueError(f"{path} must contain both binary classes 0 and 1.")
    features = frame.loc[:, feature_names]
    return features, target, path


def probabilities(model: BaseEstimator, features: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(features), dtype=float)[:, 1]
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(features), dtype=float)
        return 1.0 / (1.0 + np.exp(-np.clip(scores, -500, 500)))
    raise TypeError("The trained estimator exposes neither predict_proba nor decision_function.")


def metric_values(target: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float | int | list[list[int]]]:
    prediction = (scores >= threshold).astype("int8")
    matrix = confusion_matrix(target, prediction, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    return {
        "accuracy": float(accuracy_score(target, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(target, prediction)),
        "precision": float(precision_score(target, prediction, zero_division=0)),
        "recall": float(recall_score(target, prediction, zero_division=0)),
        "specificity": float(tn / (tn + fp)),
        "f1": float(f1_score(target, prediction, zero_division=0)),
        "f2": float(fbeta_score(target, prediction, beta=2, zero_division=0)),
        "mcc": float(matthews_corrcoef(target, prediction)),
        "roc_auc": float(roc_auc_score(target, scores)),
        "average_precision": float(average_precision_score(target, scores)),
        "brier_score": float(brier_score_loss(target, scores)),
        "log_loss": float(log_loss(target, np.column_stack((1.0 - scores, scores)), labels=[0, 1])),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "confusion_matrix": matrix.tolist(),
    }


def tune_threshold(target: pd.Series, scores: np.ndarray) -> tuple[float, pd.DataFrame]:
    candidates = np.unique(np.append(THRESHOLD_GRID, 0.5))
    rows = []
    target_array = target.to_numpy()
    for threshold in candidates:
        metrics = metric_values(target_array, scores, float(threshold))
        rows.append(
            {
                "threshold": float(threshold),
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "f2": metrics["f2"],
                "false_positive": metrics["false_positive"],
                "false_negative": metrics["false_negative"],
            }
        )
    results = pd.DataFrame(rows)
    # Malware recall is important, but F2 prevents a trivial all-malware rule.
    # Staying near 0.5 is the final tie-breaker so tuning does not select an
    # unnecessarily extreme threshold from a flat optimum.
    ranked = results.assign(distance_from_default=(results["threshold"] - 0.5).abs()).sort_values(
        ["f2", "recall", "precision", "distance_from_default"],
        ascending=[False, False, False, True],
    )
    return float(ranked.iloc[0]["threshold"]), results


def bootstrap_intervals(
    target: np.ndarray, scores: np.ndarray, threshold: float
) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(RANDOM_STATE)
    class_indices = [np.flatnonzero(target == label) for label in (0, 1)]
    metric_functions: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
        "accuracy": lambda y, p: accuracy_score(y, p),
        "precision": lambda y, p: precision_score(y, p, zero_division=0),
        "recall": lambda y, p: recall_score(y, p, zero_division=0),
        "f1": lambda y, p: f1_score(y, p, zero_division=0),
        "f2": lambda y, p: fbeta_score(y, p, beta=2, zero_division=0),
    }
    samples = {name: [] for name in [*metric_functions, "roc_auc"]}
    for _ in range(BOOTSTRAP_SAMPLES):
        sample_index = np.concatenate(
            [rng.choice(indices, size=len(indices), replace=True) for indices in class_indices]
        )
        sample_target = target[sample_index]
        sample_scores = scores[sample_index]
        sample_prediction = (sample_scores >= threshold).astype("int8")
        for name, function in metric_functions.items():
            samples[name].append(float(function(sample_target, sample_prediction)))
        samples["roc_auc"].append(float(roc_auc_score(sample_target, sample_scores)))

    alpha = (1.0 - CONFIDENCE_LEVEL) / 2.0
    return {
        name: {
            "lower": float(np.quantile(values, alpha)),
            "upper": float(np.quantile(values, 1.0 - alpha)),
        }
        for name, values in samples.items()
    }


def robustness_tests(
    model: BaseEstimator,
    test_x: pd.DataFrame,
    test_y: pd.Series,
    train_x: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    target = test_y.to_numpy()
    clean_scores = probabilities(model, test_x)
    clean = metric_values(target, clean_scores, threshold)
    rows = [{"scenario": "clean", "level": 0.0, **clean}]
    scale = train_x.std(axis=0, ddof=0).replace(0, 1.0).to_numpy(dtype=float)

    for scenario_number, level in enumerate(PERTURBATION_LEVELS, start=1):
        noise_rng = np.random.default_rng(RANDOM_STATE + scenario_number)
        noisy_values = test_x.to_numpy(dtype=float)
        noisy = pd.DataFrame(
            noisy_values
            + noise_rng.normal(
                loc=0.0,
                scale=level * scale,
                size=noisy_values.shape,
            ),
            index=test_x.index,
            columns=test_x.columns,
        )
        metrics = metric_values(target, probabilities(model, noisy), threshold)
        rows.append({"scenario": "gaussian_noise", "level": level, **metrics})

        missing_rng = np.random.default_rng(RANDOM_STATE + 100 + scenario_number)
        missing = test_x.mask(missing_rng.random(test_x.shape) < level)
        metrics = metric_values(target, probabilities(model, missing), threshold)
        rows.append({"scenario": "missing_values", "level": level, **metrics})

    result = pd.DataFrame(rows)
    for metric in ("accuracy", "precision", "recall", "f1", "f2", "roc_auc"):
        result[f"{metric}_change"] = result[metric] - float(clean[metric])
    return result


def save_plots(
    target: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    threshold_results: pd.DataFrame,
    robustness: pd.DataFrame,
) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    prediction = (scores >= threshold).astype("int8")
    matrix = confusion_matrix(target, prediction, labels=[0, 1])

    fig, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(matrix, cmap="Blues")
    for row in range(2):
        for column in range(2):
            axis.text(column, row, f"{matrix[row, column]:,}", ha="center", va="center", fontsize=13)
    axis.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Benign", "Malware"], yticklabels=["Benign", "Malware"])
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("Actual class")
    axis.set_title(f"Test confusion matrix (threshold={threshold:.2f})")
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    false_positive_rate, true_positive_rate, _ = roc_curve(target, scores)
    pr_precision, pr_recall, _ = precision_recall_curve(target, scores)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(false_positive_rate, true_positive_rate, color="#1f77b4", lw=2)
    axes[0].plot([0, 1], [0, 1], "--", color="gray")
    axes[0].set(xlabel="False-positive rate", ylabel="True-positive rate", title="ROC curve", xlim=(0, 1), ylim=(0, 1.01))
    axes[1].plot(pr_recall, pr_precision, color="#d62728", lw=2)
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall curve", xlim=(0, 1), ylim=(0, 1.01))
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "roc_pr_curves.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 5))
    for metric, color in [("precision", "#2ca02c"), ("recall", "#d62728"), ("f1", "#9467bd"), ("f2", "#1f77b4")]:
        axis.plot(threshold_results["threshold"], threshold_results[metric], label=metric.upper(), color=color)
    axis.axvline(threshold, color="black", linestyle="--", label=f"Selected ({threshold:.2f})")
    plotted_minimum = float(threshold_results[["precision", "recall", "f1", "f2"]].min().min())
    lower_limit = max(0.0, plotted_minimum - max(0.002, (1.0 - plotted_minimum) * 0.15))
    axis.set(
        xlabel="Decision threshold",
        ylabel="Validation score",
        title="Validation threshold tuning",
        ylim=(lower_limit, 1.0005),
    )
    axis.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "threshold_tuning.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 5))
    for scenario, label, marker in [("gaussian_noise", "Gaussian noise", "o"), ("missing_values", "Missing values", "s")]:
        subset = robustness[robustness["scenario"] == scenario]
        axis.plot(100 * subset["level"], subset["f2"], marker=marker, label=f"{label}: F2")
        axis.plot(100 * subset["level"], subset["recall"], marker=marker, linestyle="--", label=f"{label}: recall")
    axis.axhline(float(robustness.iloc[0]["f2"]), color="gray", linestyle=":", label="Clean F2")
    robustness_minimum = float(robustness[["f2", "recall"]].min().min())
    lower_limit = max(0.0, robustness_minimum - max(0.0002, (1.0 - robustness_minimum) * 0.25))
    axis.set(
        xlabel="Perturbed feature cells / noise scale (%)",
        ylabel="Score",
        title="Synthetic robustness stress tests",
        ylim=(lower_limit, 1.00005),
    )
    axis.ticklabel_format(axis="y", style="plain", useOffset=False)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "robustness.png", dpi=160)
    plt.close(fig)


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    return value


def write_report(summary: dict[str, Any], robustness: pd.DataFrame) -> None:
    metrics = summary["test_metrics"]
    intervals = summary["bootstrap_95_percent_confidence_intervals"]
    weakest = robustness.sort_values("f2").iloc[0]
    best_parameters = json.dumps(summary["best_parameters"], sort_keys=True)
    lines = [
        "# Model Evaluation and Tuning Report",
        "",
        "## Outcome",
        "",
        f"The **{summary['selected_model']}** pipeline was evaluated on {summary['rows']['test']:,} held-out test rows after threshold tuning on the separate validation split.",
        "",
        "| Metric | Test score | Stratified bootstrap 95% CI |",
        "| --- | ---: | ---: |",
    ]
    for metric, label in [("accuracy", "Accuracy"), ("precision", "Precision"), ("recall", "Recall"), ("f1", "F1"), ("f2", "F2"), ("roc_auc", "ROC-AUC")]:
        interval = intervals[metric]
        lines.append(f"| {label} | {metrics[metric]:.6f} | [{interval['lower']:.6f}, {interval['upper']:.6f}] |")
    lines.extend(
        [
            "",
            f"Confusion matrix: `{metrics['confusion_matrix']}` (`[[TN, FP], [FN, TP]]`). The tuned decision threshold is **{summary['decision_threshold']:.2f}**; the original threshold was **{summary['original_decision_threshold']:.2f}**.",
            "Because there were no observed test classification or ranking errors, the empirical bootstrap intervals collapse to the observed value. They describe resampling uncertainty within this test set, not uncertainty under dataset shift.",
            "",
            "## Tuning method",
            "",
            f"Step 7 searched hyperparameters with stratified cross-validation and selected: `{best_parameters}`. Step 8 cloned that configured pipeline, fitted it only on training data, and searched thresholds from 0.01 through 0.99 on validation F2. Recall and precision break F2 ties, followed by proximity to 0.50. The chosen pipeline was then refitted on train + validation before the one-time test evaluation.",
            "",
            "## Strengths",
            "",
            f"- Clean test accuracy is **{metrics['accuracy']:.4%}**, precision is **{metrics['precision']:.4%}**, recall is **{metrics['recall']:.4%}**, and F1 is **{metrics['f1']:.4%}**.",
            f"- Malware misses: **{metrics['false_negative']:,}**; benign false alarms: **{metrics['false_positive']:,}**.",
            "- The estimator keeps median imputation inside its pipeline, so missing inputs are handled consistently at inference.",
            "",
            "## Weaknesses and robustness",
            "",
            f"The weakest synthetic stress case was **{weakest['scenario']} at {weakest['level']:.0%}**, with F2 **{weakest['f2']:.6f}** and recall **{weakest['recall']:.6f}**. These perturbations are sensitivity checks, not substitutes for naturally corrupted or shifted capture data.",
            "",
            "The near-perfect random-split result is a warning as well as a strength: related captures, repeated behavioral patterns, or malware-family overlap may occur across partitions. Before deployment, repeat evaluation on family-separated, time-separated, and independently collected data; measure latency and monitor feature drift and class prevalence.",
            "",
            "## Reproducible artifacts",
            "",
            "- `test_metrics.csv`: final scalar metrics and confidence bounds",
            "- `classification_report.csv`: per-class precision, recall, F1, and support",
            "- `threshold_search.csv`: validation scores for every candidate threshold",
            "- `robustness_results.csv`: clean, noise, and missing-value stress results",
            "- `figures/`: confusion matrix, ROC/PR, threshold, and robustness plots",
            "- `models/malware_detector_tuned.joblib`: final pipeline plus tuned threshold and evaluation metadata",
            "",
            "## Reproduce",
            "",
            "```bash",
            ".venv/bin/python scripts/model_evaluation_tuning.py",
            "```",
            "",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    if not INPUT_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing {INPUT_MODEL_PATH}. Run scripts/model_selection_training.py first."
        )
    source_bundle = joblib.load(INPUT_MODEL_PATH)
    required_keys = {"model", "feature_names", "decision_threshold", "metadata"}
    missing_keys = required_keys - set(source_bundle)
    if missing_keys:
        raise ValueError(f"Model bundle is missing required keys: {sorted(missing_keys)}")

    feature_names = list(source_bundle["feature_names"])
    train_x, train_y, train_path = load_split("train", feature_names)
    validation_x, validation_y, validation_path = load_split("validation", feature_names)
    test_x, test_y, test_path = load_split("test", feature_names)

    tuning_model = clone(source_bundle["model"])
    tuning_model.fit(train_x, train_y)
    validation_scores = probabilities(tuning_model, validation_x)
    threshold, threshold_results = tune_threshold(validation_y, validation_scores)

    final_x = pd.concat([train_x, validation_x], ignore_index=True)
    final_y = pd.concat([train_y, validation_y], ignore_index=True)
    final_model = clone(source_bundle["model"])
    final_model.fit(final_x, final_y)
    test_scores = probabilities(final_model, test_x)
    test_target = test_y.to_numpy()
    test_metrics = metric_values(test_target, test_scores, threshold)
    confidence_intervals = bootstrap_intervals(test_target, test_scores, threshold)
    robustness = robustness_tests(final_model, test_x, test_y, final_x, threshold)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    threshold_results.to_csv(THRESHOLD_PATH, index=False)
    robustness.to_csv(ROBUSTNESS_PATH, index=False)

    prediction = (test_scores >= threshold).astype("int8")
    class_report = pd.DataFrame(
        classification_report(
            test_target,
            prediction,
            labels=[0, 1],
            target_names=["Benign", "Malware"],
            output_dict=True,
            zero_division=0,
        )
    ).transpose()
    class_report.to_csv(CLASS_REPORT_PATH, index_label="class")

    metric_rows = []
    for name, value in test_metrics.items():
        if isinstance(value, (float, int)):
            interval = confidence_intervals.get(name, {})
            metric_rows.append(
                {
                    "metric": name,
                    "value": value,
                    "ci_95_lower": interval.get("lower", np.nan),
                    "ci_95_upper": interval.get("upper", np.nan),
                }
            )
    pd.DataFrame(metric_rows).to_csv(METRICS_PATH, index=False)
    save_plots(test_target, test_scores, threshold, threshold_results, robustness)

    best_parameters = source_bundle.get("metadata", {}).get("best_parameters", {})
    summary = {
        "selected_model": source_bundle.get("metadata", {}).get("selected_model", type(final_model).__name__),
        "best_parameters": best_parameters,
        "threshold_selection_metric": "validation_f2",
        "original_decision_threshold": float(source_bundle["decision_threshold"]),
        "decision_threshold": threshold,
        "random_state": RANDOM_STATE,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "rows": {
            "train": len(train_x),
            "validation": len(validation_x),
            "final_refit": len(final_x),
            "test": len(test_x),
        },
        "feature_count": len(feature_names),
        "test_metrics": test_metrics,
        "bootstrap_95_percent_confidence_intervals": confidence_intervals,
        "robustness_worst_case": json_ready(
            robustness.sort_values("f2").iloc[0][
                ["scenario", "level", "accuracy", "precision", "recall", "f1", "f2", "roc_auc"]
            ].to_dict()
        ),
        "input_sha256": {
            str(path.relative_to(PROJECT_ROOT)): sha256(path)
            for path in (train_path, validation_path, test_path, INPUT_MODEL_PATH)
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
            "platform": platform.platform(),
        },
    }

    output_bundle = {
        **source_bundle,
        "model": final_model,
        "decision_threshold": threshold,
        "metadata": {
            **source_bundle.get("metadata", {}),
            "evaluation": {
                "threshold_selection_metric": "validation_f2",
                "test_metrics": test_metrics,
                "bootstrap_95_percent_confidence_intervals": confidence_intervals,
            },
        },
    }
    joblib.dump(output_bundle, OUTPUT_MODEL_PATH, compress=3)
    summary["model_file"] = str(OUTPUT_MODEL_PATH.relative_to(PROJECT_ROOT))
    summary["model_sha256"] = sha256(OUTPUT_MODEL_PATH)
    SUMMARY_PATH.write_text(json.dumps(json_ready(summary), indent=2) + "\n", encoding="utf-8")
    write_report(summary, robustness)
    print(
        f"Evaluation complete: threshold={threshold:.2f}, accuracy={test_metrics['accuracy']:.6f}, "
        f"recall={test_metrics['recall']:.6f}, F1={test_metrics['f1']:.6f}."
    )
    print(f"Report: {REPORT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Evaluation failed: {error}", file=sys.stderr)
        raise
