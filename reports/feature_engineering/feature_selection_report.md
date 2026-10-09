# Feature Engineering and Selection Report

## Outcome

The process added **25 domain features** to the 52 original measurements, then selected **25 features** using only the training partition. Of those, **10 are engineered features**. Correlation pruning removed **29 redundant candidates** at an absolute training correlation threshold of 0.95.

The exported train, validation, and test files contain only the selected predictors and `Class`. Their separation must be preserved during modeling.

## Leakage controls

- A stratified 64%/16%/20% train/validation/test split was created before target-aware selection.
- Ratios are deterministic row-level transformations and do not use `Class` or dataset-wide statistics.
- Mutual information and correlation pruning were fitted on the training partition only.
- Scaling and median-imputation safeguards were fitted inside each training-only logistic-regression pipeline.
- The raw and cleansed source files were not modified.

## Selected features

1. `derived.process_service_rate` (MI=0.6865, engineered)
2. `derived.driver_service_rate` (MI=0.6857, engineered)
3. `derived.active_service_rate` (MI=0.6853, engineered)
4. `svcscan.nservices` (MI=0.6847, original)
5. `derived.shared_process_service_rate` (MI=0.6843, engineered)
6. `svcscan.kernel_drivers` (MI=0.6750, original)
7. `derived.dlls_per_process_recomputed` (MI=0.6729, engineered)
8. `derived.mutants_per_process` (MI=0.6721, engineered)
9. `derived.sections_per_process` (MI=0.6699, engineered)
10. `handles.avg_handles_per_proc` (MI=0.6573, original)
11. `handles.nmutant` (MI=0.6489, original)
12. `handles.nevent` (MI=0.6411, original)
13. `handles.nhandles` (MI=0.6298, original)
14. `handles.nkey` (MI=0.6275, original)
15. `handles.nsemaphore` (MI=0.6185, original)
16. `handles.ntimer` (MI=0.6023, original)
17. `pslist.avg_threads` (MI=0.5909, original)
18. `derived.commit_charge_per_injection` (MI=0.5871, engineered)
19. `handles.nfile` (MI=0.5867, original)
20. `ldrmodules.not_in_load` (MI=0.5836, original)
21. `derived.unloaded_module_rate` (MI=0.5594, engineered)
22. `malfind.uniqueInjections` (MI=0.5575, original)
23. `ldrmodules.not_in_load_avg` (MI=0.5466, original)
24. `derived.uninitialized_module_rate` (MI=0.5283, engineered)
25. `svcscan.nactive` (MI=0.4746, original)

## Held-out benchmark

This comparison uses the same split and the same regularized logistic-regression estimator. The baseline uses all 52 original features; the selected model uses the engineered-and-pruned set.

| Feature set | Split | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Original 52 | validation | 0.9988 | 0.9987 | 0.9989 | 0.9988 | 0.99994 |
| Original 52 | test | 0.9994 | 0.9997 | 0.9991 | 0.9994 | 0.99999 |
| Selected 25 | validation | 0.9987 | 0.9985 | 0.9989 | 0.9987 | 0.99971 |
| Selected 25 | test | 0.9991 | 0.9998 | 0.9983 | 0.9990 | 0.99996 |

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
- Input SHA-256: `f5e81dc47faa0d5204091582155f9f7692a6f9ddbea7da70b25ae4d1b3a3bfbd`
- Random seed: 42
- Script: `scripts/feature_engineering.py`
- Selected split files: `data/processed/*_selected.csv.gz`

Run from the project root:

```bash
.venv/bin/python scripts/feature_engineering.py
```
