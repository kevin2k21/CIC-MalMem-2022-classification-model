"""Run exploratory data analysis on the cleansed CIC-MalMem-2022 dataset.

The script treats ``data/cleansed_data.csv`` as read-only and writes the EDA
report, machine-readable summaries, tables, and charts under ``reports/eda``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "cleansed_data.csv"
AUDIT_PATH = PROJECT_ROOT / "reports" / "data_cleaning_audit.json"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "eda"
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
REPORT_PATH = OUTPUT_DIR / "eda_report.md"
SUMMARY_PATH = OUTPUT_DIR / "eda_summary.json"
RANDOM_STATE = 42


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_plot_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 160,
            "axes.titleweight": "bold",
            "axes.labelcolor": "#273142",
            "text.color": "#273142",
            "font.size": 10,
        }
    )


def save_figure(filename: str) -> None:
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / filename, bbox_inches="tight", facecolor="white")
    plt.close()


def calculate_feature_statistics(features: pd.DataFrame) -> pd.DataFrame:
    quantiles = features.quantile([0.25, 0.50, 0.75]).T
    stats = pd.DataFrame(index=features.columns)
    stats.index.name = "feature"
    stats["dtype"] = features.dtypes.astype(str)
    stats["missing_count"] = features.isna().sum()
    stats["unique_count"] = features.nunique(dropna=False)
    stats["mean"] = features.mean()
    stats["std"] = features.std(ddof=1)
    stats["min"] = features.min()
    stats["q1"] = quantiles[0.25]
    stats["median"] = quantiles[0.50]
    stats["q3"] = quantiles[0.75]
    stats["max"] = features.max()
    stats["skewness"] = features.skew()
    stats["excess_kurtosis"] = features.kurt()
    stats["zero_rate"] = features.eq(0).mean()

    iqr = stats["q3"] - stats["q1"]
    lower = stats["q1"] - 1.5 * iqr
    upper = stats["q3"] + 1.5 * iqr
    outlier_rates = {}
    for feature in features.columns:
        outliers = (features[feature] < lower[feature]) | (
            features[feature] > upper[feature]
        )
        outlier_rates[feature] = float(outliers.mean())
    stats["iqr_outlier_rate"] = pd.Series(outlier_rates)
    return stats.reset_index()


def calculate_class_comparison(
    features: pd.DataFrame, target: pd.Series
) -> pd.DataFrame:
    benign = features.loc[target.eq(0)]
    malware = features.loc[target.eq(1)]
    rows: list[dict[str, object]] = []

    for feature in features.columns:
        benign_values = benign[feature]
        malware_values = malware[feature]
        pooled_std = np.sqrt(
            (
                (len(benign_values) - 1) * benign_values.var(ddof=1)
                + (len(malware_values) - 1) * malware_values.var(ddof=1)
            )
            / (len(benign_values) + len(malware_values) - 2)
        )
        standardized_mean_difference = (
            (malware_values.mean() - benign_values.mean()) / pooled_std
            if pooled_std > 0
            else 0.0
        )
        correlation = float(features[feature].corr(target))
        raw_auc = float(roc_auc_score(target, features[feature]))
        directional_auc = max(raw_auc, 1.0 - raw_auc)
        rows.append(
            {
                "feature": feature,
                "benign_mean": benign_values.mean(),
                "malware_mean": malware_values.mean(),
                "benign_median": benign_values.median(),
                "malware_median": malware_values.median(),
                "mean_difference_malware_minus_benign": malware_values.mean()
                - benign_values.mean(),
                "standardized_mean_difference": standardized_mean_difference,
                "point_biserial_correlation": correlation,
                "absolute_correlation": abs(correlation),
                "univariate_auc": raw_auc,
                "direction_adjusted_auc": directional_auc,
                "higher_in": "Malware" if correlation >= 0 else "Benign",
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["absolute_correlation", "direction_adjusted_auc"], ascending=False
    )


def calculate_correlation_pairs(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    correlation = features.corr(method="pearson")
    mask = np.triu(np.ones(correlation.shape, dtype=bool), k=1)
    pairs = (
        correlation.where(mask)
        .rename_axis(index="feature_1", columns="feature_2")
        .stack()
        .rename("correlation")
        .reset_index()
        .dropna(subset=["correlation"])
    )
    pairs["absolute_correlation"] = pairs["correlation"].abs()
    pairs = pairs.sort_values("absolute_correlation", ascending=False)
    return correlation, pairs


def plot_class_distribution(target: pd.Series) -> None:
    counts = target.value_counts().sort_index()
    labels = ["Benign (0)", "Malware (1)"]
    colors = ["#2A6FBB", "#D04A35"]
    plt.figure(figsize=(7.2, 4.5))
    axis = sns.barplot(x=labels, y=counts.values, hue=labels, palette=colors, legend=False)
    axis.set_title("Class distribution")
    axis.set_xlabel("")
    axis.set_ylabel("Records")
    total = len(target)
    for index, count in enumerate(counts.values):
        axis.text(index, count + total * 0.008, f"{count:,}\n({count / total:.1%})", ha="center")
    axis.set_ylim(0, counts.max() * 1.14)
    save_figure("class_distribution.png")


def plot_top_associations(class_comparison: pd.DataFrame, top_n: int = 15) -> None:
    data = class_comparison.head(top_n).sort_values("point_biserial_correlation")
    colors = np.where(data["point_biserial_correlation"] >= 0, "#D04A35", "#2A6FBB")
    plt.figure(figsize=(10, 7))
    axis = plt.gca()
    axis.barh(data["feature"], data["point_biserial_correlation"], color=colors)
    axis.axvline(0, color="#555555", linewidth=0.8)
    axis.set_title("Features most associated with malware class")
    axis.set_xlabel("Point-biserial correlation with Class (1 = Malware)")
    axis.set_ylabel("")
    save_figure("top_class_associations.png")


def plot_correlation_heatmap(
    feature_correlation: pd.DataFrame,
    class_comparison: pd.DataFrame,
    top_n: int = 16,
) -> None:
    top_features = class_comparison.head(top_n)["feature"].tolist()
    matrix = feature_correlation.loc[top_features, top_features]
    plt.figure(figsize=(13, 11))
    sns.heatmap(
        matrix,
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.25,
        cbar_kws={"label": "Pearson correlation", "shrink": 0.8},
    )
    plt.title("Correlation among the strongest class-associated features")
    save_figure("correlation_heatmap.png")


def plot_feature_distributions(
    frame: pd.DataFrame, class_comparison: pd.DataFrame, top_n: int = 6
) -> None:
    top_features = class_comparison.head(top_n)["feature"].tolist()
    sample_size = min(12_000, len(frame))
    sample = frame.groupby("Class", group_keys=False).sample(
        n=min(sample_size // 2, frame["Class"].value_counts().min()),
        random_state=RANDOM_STATE,
    )
    long_parts = []
    for feature in top_features:
        values = sample[feature]
        median = values.median()
        iqr = values.quantile(0.75) - values.quantile(0.25)
        scale = iqr if iqr > 0 else values.std(ddof=1)
        scale = scale if scale > 0 else 1.0
        robust_scaled = ((values - median) / scale).clip(-5, 5)
        long_parts.append(
            pd.DataFrame(
                {
                    "feature": feature,
                    "robust_scaled_value": robust_scaled,
                    "Class": sample["Class"].map({0: "Benign", 1: "Malware"}),
                }
            )
        )
    long_data = pd.concat(long_parts, ignore_index=True)
    grid = sns.catplot(
        data=long_data,
        x="robust_scaled_value",
        y="Class",
        hue="Class",
        col="feature",
        col_wrap=2,
        kind="box",
        height=2.5,
        aspect=1.8,
        palette={"Benign": "#2A6FBB", "Malware": "#D04A35"},
        legend=False,
        showfliers=False,
        sharex=False,
    )
    grid.set_axis_labels("Robust-scaled value", "")
    grid.set_titles("{col_name}")
    grid.figure.suptitle("Class distributions for the strongest features", y=1.02, fontweight="bold")
    grid.figure.tight_layout()
    grid.figure.savefig(
        FIGURE_DIR / "top_feature_distributions.png",
        bbox_inches="tight",
        facecolor="white",
        dpi=160,
    )
    plt.close(grid.figure)


def plot_outlier_rates(feature_stats: pd.DataFrame, top_n: int = 15) -> None:
    data = feature_stats.nlargest(top_n, "iqr_outlier_rate").sort_values("iqr_outlier_rate")
    plt.figure(figsize=(10, 7))
    axis = plt.gca()
    axis.barh(data["feature"], data["iqr_outlier_rate"] * 100, color="#7B61A8")
    axis.set_title("Features with the highest IQR outlier rates")
    axis.set_xlabel("Records outside 1.5 × IQR (%)")
    axis.set_ylabel("")
    save_figure("outlier_rates.png")


def plot_pca(features: pd.DataFrame, target: pd.Series) -> dict[str, float]:
    scaled = StandardScaler().fit_transform(features)
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coordinates = pca.fit_transform(scaled)
    plot_frame = pd.DataFrame(
        {"PC1": coordinates[:, 0], "PC2": coordinates[:, 1], "Class": target}
    )
    if len(plot_frame) > 8_000:
        plot_frame = plot_frame.groupby("Class", group_keys=False).sample(
            n=4_000, random_state=RANDOM_STATE
        )
    plot_frame["Class"] = plot_frame["Class"].map({0: "Benign", 1: "Malware"})

    plt.figure(figsize=(8.5, 6.5))
    sns.scatterplot(
        data=plot_frame,
        x="PC1",
        y="PC2",
        hue="Class",
        palette={"Benign": "#2A6FBB", "Malware": "#D04A35"},
        alpha=0.32,
        s=13,
        linewidth=0,
    )
    plt.title("PCA projection of standardized features")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    plt.legend(title="Class", frameon=True)
    save_figure("pca_projection.png")
    return {
        "pc1_explained_variance_ratio": float(pca.explained_variance_ratio_[0]),
        "pc2_explained_variance_ratio": float(pca.explained_variance_ratio_[1]),
        "first_two_components_explained_variance_ratio": float(
            pca.explained_variance_ratio_.sum()
        ),
    }


def markdown_table(data: pd.DataFrame, columns: list[str], rows: int = 10) -> str:
    view = data.loc[:, columns].head(rows).copy()
    for column in view.select_dtypes(include="number").columns:
        view[column] = view[column].map(lambda value: f"{value:.3f}")
    headers = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = "\n".join(
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in view.itertuples(index=False, name=None)
    )
    return "\n".join([headers, divider, body])


def write_report(
    frame: pd.DataFrame,
    feature_stats: pd.DataFrame,
    class_comparison: pd.DataFrame,
    correlation_pairs: pd.DataFrame,
    pca_summary: dict[str, float],
    audit: dict[str, object],
) -> None:
    counts = frame["Class"].value_counts().sort_index()
    shares = counts / len(frame)
    highly_correlated = correlation_pairs[correlation_pairs["absolute_correlation"] >= 0.90]
    top_outliers = feature_stats.nlargest(5, "iqr_outlier_rate")
    top_features = class_comparison.head(5)

    top_feature_sentences = []
    for row in top_features.itertuples(index=False):
        top_feature_sentences.append(
            f"`{row.feature}` (r = {row.point_biserial_correlation:.3f}, "
            f"direction-adjusted AUC = {row.direction_adjusted_auc:.3f}, higher in {row.higher_in.lower()})"
        )

    report = f"""# Exploratory Data Analysis: Cleansed CIC-MalMem-2022 Data

## Executive summary

- The cleansed dataset contains **{len(frame):,} records**, **{frame.shape[1] - 1} numeric features**, and one binary target. There are **{int(frame.isna().sum().sum()):,} missing cells**, **{int(frame.duplicated().sum()):,} duplicate rows**, and no non-finite numeric values.
- The classes are nearly balanced: **{counts.get(0, 0):,} benign ({shares.get(0, 0):.1%})** and **{counts.get(1, 0):,} malware ({shares.get(1, 0):.1%})**. Class weighting is unlikely to be essential for the first baseline, but stratified splitting remains appropriate.
- The strongest univariate class associations are {", ".join(top_feature_sentences)}. These features are useful candidates for baseline modeling, but the results do not establish causal importance.
- The best single-feature ranking is unusually strong. Treat this as a reason to test for collection artifacts and related samples across data splits, not as proof that deployment performance will be equally high.
- **{len(highly_correlated):,} feature pairs** have an absolute Pearson correlation of at least 0.90. This indicates substantial redundancy and supports testing regularization, correlation-based pruning, or tree-based models.
- The first two principal components explain **{pca_summary['first_two_components_explained_variance_ratio']:.1%}** of standardized feature variance. The 2-D projection is exploratory and should not be treated as a measure of final classification performance.
- IQR rules flag the highest apparent outlier rates in {", ".join(f"`{row.feature}` ({row.iqr_outlier_rate:.1%})" for row in top_outliers.itertuples(index=False))}. In cybersecurity telemetry, extreme values may carry attack signal, so they should not be removed automatically.

## Dataset structure and quality

| Check | Result |
| --- | ---: |
| Rows | {len(frame):,} |
| Predictive features | {frame.shape[1] - 1} |
| Integer features | {int(frame.drop(columns='Class').dtypes.eq('int64').sum())} |
| Floating-point features | {int(frame.drop(columns='Class').dtypes.eq('float64').sum())} |
| Missing cells | {int(frame.isna().sum().sum()):,} |
| Infinite numeric cells | {int(np.isinf(frame.select_dtypes(include='number').to_numpy()).sum()):,} |
| Exact duplicate rows | {int(frame.duplicated().sum()):,} |

The cleaning audit maps `Class = 0` to benign and `Class = 1` to malware. It also records removal of leakage (`Category`), constant features, and duplicates. Scaling remains deferred until after the train/test split.

![Class distribution](figures/class_distribution.png)

## Class relationships

The table ranks features by absolute point-biserial correlation with the binary target. Direction-adjusted AUC describes the best one-feature ranking direction and is not a cross-validated model score.

{markdown_table(class_comparison, ['feature', 'point_biserial_correlation', 'standardized_mean_difference', 'direction_adjusted_auc', 'higher_in'])}

![Top class associations](figures/top_class_associations.png)

![Top feature distributions](figures/top_feature_distributions.png)

## Feature relationships and redundancy

The strongest feature-to-feature relationships are shown below. Highly correlated variables can make linear coefficients unstable and may duplicate signal.

{markdown_table(correlation_pairs, ['feature_1', 'feature_2', 'correlation', 'absolute_correlation'])}

![Correlation heatmap](figures/correlation_heatmap.png)

## Distribution shape and outliers

The IQR outlier rate is a diagnostic, not a deletion rule. Review extreme values against domain expectations and fit any clipping or transformation using training data only.

{markdown_table(top_outliers, ['feature', 'skewness', 'zero_rate', 'iqr_outlier_rate'], rows=5)}

![Outlier rates](figures/outlier_rates.png)

## Multivariate structure

PCA was fitted to standardized features for visualization. PC1 explains **{pca_summary['pc1_explained_variance_ratio']:.1%}** and PC2 explains **{pca_summary['pc2_explained_variance_ratio']:.1%}** of the variance.

![PCA projection](figures/pca_projection.png)

## Modeling implications

1. Use a stratified train/validation/test split and fit scaling, imputation safeguards, feature selection, and any outlier handling on training data only.
2. Start with an interpretable regularized logistic-regression baseline, then compare tree ensembles that can capture nonlinear interactions.
3. Evaluate correlation pruning or regularization because the dataset contains strongly redundant feature groups.
4. Report precision, recall, F1, ROC-AUC, and the confusion matrix. For malware detection, choose the operating threshold according to the cost of false negatives and false positives.
5. Check split strategy against collection or malware-family grouping if those identifiers are available outside this leakage-safe table. Random row splits can overstate generalization when closely related samples occur across folds.
6. Investigate the near-perfect one-feature AUC before trusting benchmark scores. Validate on a time-separated, family-separated, or independently collected test set when possible.

## Reproducibility

- Input: `data/cleansed_data.csv`
- Cleaning audit: `reports/data_cleaning_audit.json`
- Script: `scripts/eda.py`
- Random seed: {RANDOM_STATE}
- Source SHA-256: `{sha256(DATA_PATH)}`
- Cleaning audit source rows: {audit.get('before', {}).get('rows', 'n/a')}

Run from the project root:

```bash
.venv/bin/python scripts/eda.py
```
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing cleansed dataset: {DATA_PATH}")
    if not AUDIT_PATH.exists():
        raise FileNotFoundError(f"Missing cleaning audit: {AUDIT_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    set_plot_style()

    source_hash_before = sha256(DATA_PATH)
    frame = pd.read_csv(DATA_PATH)
    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))

    if "Class" not in frame.columns:
        raise ValueError("Expected a Class target column.")
    if set(frame["Class"].dropna().unique()) != {0, 1}:
        raise ValueError("Class must contain exactly the encoded labels 0 and 1.")
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in frame.dtypes):
        raise ValueError("EDA expects the cleansed dataset to contain only numeric columns.")

    features = frame.drop(columns="Class")
    target = frame["Class"].astype("int8")

    feature_stats = calculate_feature_statistics(features)
    class_comparison = calculate_class_comparison(features, target)
    feature_correlation, correlation_pairs = calculate_correlation_pairs(features)

    feature_stats.to_csv(TABLE_DIR / "feature_statistics.csv", index=False)
    class_comparison.to_csv(TABLE_DIR / "class_comparison.csv", index=False)
    correlation_pairs.to_csv(TABLE_DIR / "feature_correlation_pairs.csv", index=False)

    plot_class_distribution(target)
    plot_top_associations(class_comparison)
    plot_correlation_heatmap(feature_correlation, class_comparison)
    plot_feature_distributions(frame, class_comparison)
    plot_outlier_rates(feature_stats)
    pca_summary = plot_pca(features, target)

    counts = target.value_counts().sort_index()
    summary = {
        "source_file": str(DATA_PATH.relative_to(PROJECT_ROOT)),
        "source_sha256": source_hash_before,
        "rows": int(frame.shape[0]),
        "columns": int(frame.shape[1]),
        "feature_count": int(features.shape[1]),
        "missing_cells": int(frame.isna().sum().sum()),
        "infinite_numeric_cells": int(np.isinf(frame.to_numpy()).sum()),
        "exact_duplicate_rows": int(frame.duplicated().sum()),
        "class_counts": {str(key): int(value) for key, value in counts.items()},
        "class_shares": {str(key): float(value / len(frame)) for key, value in counts.items()},
        "high_correlation_pair_count_abs_ge_0_90": int(
            correlation_pairs["absolute_correlation"].ge(0.90).sum()
        ),
        "top_class_associations": class_comparison.head(10).to_dict(orient="records"),
        "pca": pca_summary,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_report(
        frame,
        feature_stats,
        class_comparison,
        correlation_pairs,
        pca_summary,
        audit,
    )

    source_hash_after = sha256(DATA_PATH)
    if source_hash_before != source_hash_after:
        raise RuntimeError("The cleansed source changed during EDA.")

    print(json.dumps(summary, indent=2))
    print(f"\nEDA report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
