from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assignment_report"
FIG = OUT / "figures"
DATASET = ROOT / "data" / "Obfuscated-MalMem2022.csv"
RANDOM_STATE = 42


def add_figure_title(axis: plt.Axes, title: str, subtitle: str | None = None) -> None:
    axis.set_title(title, loc="left", fontsize=14, fontweight="bold", pad=14)
    if subtitle:
        axis.text(0, 1.015, subtitle, transform=axis.transAxes, fontsize=9, color="#555555")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(DATASET)
    raw.columns = raw.columns.str.strip()
    before = {
        "rows": int(len(raw)),
        "columns": int(raw.shape[1]),
        "missing_values": int(raw.isna().sum().sum()),
        "duplicate_rows": int(raw.duplicated().sum()),
        "class_counts": raw["Class"].value_counts().to_dict(),
    }

    frame = raw.replace([np.inf, -np.inf], np.nan).drop_duplicates().copy()
    target = frame["Class"].map({"Benign": 0, "Malware": 1}).astype("int8")
    features = frame.drop(columns=["Class", "Category"], errors="ignore")
    constant_columns = [column for column in features if features[column].nunique(dropna=False) <= 1]
    features = features.drop(columns=constant_columns)

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("selector", SelectKBest(score_func=f_classif, k=10)),
            (
                "model",
                DecisionTreeClassifier(
                    criterion="gini",
                    max_depth=8,
                    min_samples_leaf=2,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    pipeline.fit(x_train, y_train)
    prediction = pipeline.predict(x_test)
    probability = pipeline.predict_proba(x_test)[:, 1]

    selector = pipeline.named_steps["selector"]
    selected_features = x_train.columns[selector.get_support()].tolist()
    feature_table = pd.DataFrame(
        {
            "rank": np.arange(1, 11),
            "feature": selected_features,
            "anova_f_score": selector.scores_[selector.get_support()],
            "p_value": selector.pvalues_[selector.get_support()],
        }
    ).sort_values("anova_f_score", ascending=False)
    feature_table["rank"] = np.arange(1, 11)
    feature_table.to_csv(OUT / "top_10_features.csv", index=False)

    matrix = confusion_matrix(y_test, prediction, labels=[0, 1])
    metrics = {
        "accuracy": float(accuracy_score(y_test, prediction)),
        "precision": float(precision_score(y_test, prediction)),
        "recall": float(recall_score(y_test, prediction)),
        "f1_score": float(f1_score(y_test, prediction)),
        "mae": float(mean_absolute_error(y_test, prediction)),
        "mse": float(mean_squared_error(y_test, prediction)),
        "roc_auc": float(roc_auc_score(y_test, probability)),
        "confusion_matrix": matrix.tolist(),
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "selected_features": selected_features,
    }
    summary = {
        "dataset": "CIC-MalMem-2022",
        "source_file": str(DATASET.relative_to(ROOT)),
        "before_cleaning": before,
        "after_cleaning": {
            "rows": int(len(frame)),
            "predictor_columns": int(features.shape[1]),
            "missing_values": int(features.isna().sum().sum()),
            "duplicates": int(frame.duplicated().sum()),
            "constant_columns_removed": constant_columns,
        },
        "split": "80:20 stratified",
        "random_state": RANDOM_STATE,
        "model": "Decision Tree Classifier",
        "model_parameters": pipeline.named_steps["model"].get_params(),
        "metrics": metrics,
        "classification_report": classification_report(
            y_test, prediction, target_names=["Benign", "Malware"], output_dict=True
        ),
    }
    (OUT / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    predictions = pd.DataFrame(
        {
            "actual": y_test.map({0: "Benign", 1: "Malware"}).to_numpy(),
            "predicted": pd.Series(prediction).map({0: "Benign", 1: "Malware"}),
            "malware_probability": probability,
        }
    )
    predictions.head(20).to_csv(OUT / "sample_predictions.csv", index=False)

    plt.style.use("seaborn-v0_8-whitegrid")
    colors = ["#2F6690", "#C94C4C"]

    fig, axis = plt.subplots(figsize=(7.4, 4.2))
    counts = target.value_counts().sort_index()
    bars = axis.bar(["Benign", "Malware"], counts.values, color=colors, width=0.55)
    add_figure_title(axis, "Class distribution after duplicate removal", "The dataset remains almost balanced")
    axis.set_ylabel("Number of records")
    axis.set_ylim(0, max(counts.values) * 1.16)
    for bar, value in zip(bars, counts.values):
        axis.text(bar.get_x() + bar.get_width() / 2, value + 300, f"{value:,}", ha="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "class_distribution.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8.4, 5.1))
    ordered = feature_table.sort_values("anova_f_score")
    axis.barh(ordered["feature"], ordered["anova_f_score"], color="#2F6690")
    add_figure_title(axis, "Top 10 features selected using ANOVA F-test", "Selection was fitted only on the training set")
    axis.set_xlabel("ANOVA F-score")
    axis.tick_params(axis="y", labelsize=8.5)
    fig.tight_layout()
    fig.savefig(FIG / "top_10_features.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(5.9, 5.0))
    display = ConfusionMatrixDisplay(matrix, display_labels=["Benign", "Malware"])
    display.plot(ax=axis, cmap="Blues", colorbar=False, values_format=",")
    add_figure_title(axis, "Confusion matrix on the test set", "Rows are actual classes and columns are predicted classes")
    fig.tight_layout()
    fig.savefig(FIG / "confusion_matrix.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    false_positive_rate, true_positive_rate, _ = roc_curve(y_test, probability)
    fig, axis = plt.subplots(figsize=(6.7, 5.0))
    axis.plot(false_positive_rate, true_positive_rate, color="#2F6690", linewidth=2.3,
              label=f"Decision Tree (AUC = {metrics['roc_auc']:.4f})")
    axis.plot([0, 1], [0, 1], linestyle="--", color="#888888", label="Random classifier")
    axis.set(xlabel="False Positive Rate", ylabel="True Positive Rate", xlim=(0, 1), ylim=(0, 1.01))
    add_figure_title(axis, "ROC curve for malware detection", "A curve close to the top-left corner shows strong separation")
    axis.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "roc_curve.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    selected_tree_features = selected_features
    fig, axis = plt.subplots(figsize=(14, 7))
    plot_tree(
        pipeline.named_steps["model"],
        feature_names=selected_tree_features,
        class_names=["Benign", "Malware"],
        filled=True,
        rounded=True,
        max_depth=3,
        fontsize=7,
        ax=axis,
    )
    axis.set_title("First three levels of the trained Decision Tree", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "decision_tree_preview.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    metric_names = ["Accuracy", "Precision", "Recall", "F1 score"]
    metric_values = [metrics["accuracy"], metrics["precision"], metrics["recall"], metrics["f1_score"]]
    fig, axis = plt.subplots(figsize=(7.4, 4.3))
    bars = axis.bar(metric_names, metric_values, color=["#2F6690", "#5B8C5A", "#C94C4C", "#7A5195"])
    axis.set_ylim(max(0, min(metric_values) - 0.01), 1.001)
    add_figure_title(axis, "Test-set performance metrics", "Scores are calculated for Malware as the positive class")
    for bar, value in zip(bars, metric_values):
        axis.text(bar.get_x() + bar.get_width() / 2, value - 0.0014, f"{value:.4%}", ha="center", va="top", color="white", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "performance_metrics.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
