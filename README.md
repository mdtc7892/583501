# AI-Driven Intrusion Detection Framework - Banking Network Security

MSc dissertation technical artefact: an end-to-end, reproducible DDoS
intrusion detection pipeline on real CICIDS2017 traffic - EDA, cleaning,
feature engineering/selection, class-imbalance handling, Random
Forest/SVM/Neural Network models (each hyperparameter-tuned), an ensemble,
a full evaluation-metric suite (with a specific focus on **false positive
rate**), and explainability via SHAP + LIME.

## Folder Structure

```
DDoS_IDS_Project/
├── README.md                  <- you are here
├── requirements.txt
├── code/                      <- the notebook + its helper module (nothing else)
│   ├── 01_AI_Intrusion_Detection_Framework.ipynb   (already executed - all outputs visible)
│   └── utils.py                                     (config, data loading, cleaning, metrics helpers)
├── dataset/                   <- real CICIDS2017 data (already included, see dataset/README.md)
│   ├── Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv   (full file, 225,745 flows)
│   ├── cicids2017_real_sample.csv                          (6,000-row stratified sample - used by default)
│   └── README.md
├── models/                    <- trained model artefacts
│   ├── random_forest.joblib
│   ├── svm.joblib
│   ├── neural_network_best.keras
│   └── feature_scaler.joblib
├── results/                   <- everything the notebook run produced, separated from code
│   ├── figures/                (every chart: EDA, ROC/PR curves, confusion matrices, SHAP, LIME, etc.)
│   ├── metrics/                (CSVs: feature stats, feature importance, tuning comparison, final metrics)
│   └── reports/                (JSON: dataset summary, cleaning report, feature selection, final summary)
└── logs/
    └── run.log                 (full run log)
```

Code, dataset, trained models, and results/testing output are each in their
own top-level folder, as requested.

## Dataset Used & How Testing Was Done (answering the client's question)

**Dataset:** CICIDS2017, specifically the `Friday-WorkingHours-Afternoon-
DDos.pcap_ISCX.csv` capture (the official BENIGN vs DDoS traffic file),
obtained from a public GitHub mirror of the Canadian Institute for
Cybersecurity's dataset (see `dataset/README.md` for the link and
citation). Full file: 225,745 flows (97,718 BENIGN / 128,027 DDoS) - this
is real, labelled network traffic, not synthetic. The notebook trains on a
stratified 6,000-row sample of it by default (same real class ratio) so the
full pipeline - hyperparameter search, ensembling, SHAP/LIME - finishes in
a practical amount of time; the full file is also included if you want to
re-run at full scale (see `dataset/README.md` for how).

**How testing was done:**
1. **Split:** stratified 70% train / 15% validation / 15% test, so the test
   set keeps the real class ratio and is never touched by resampling.
2. **Feature engineering:** variance-threshold filter, correlation-based
   redundancy removal, then a feature kept only if it ranks highly by
   *both* Random Forest importance and mutual information (cross-checked
   further with RFE) - reduces the risk of one spurious feature driving the
   whole model.
3. **Class imbalance:** four strategies (class-weighting, random
   oversampling, SMOTE, random undersampling) are compared by 5-fold
   cross-validated F1 on the training split only, and the best one is
   selected automatically.
4. **Models:** Random Forest and SVM (each hyperparameter-tuned via
   `RandomizedSearchCV`), a Keras neural network (early stopping + LR
   schedule), and a soft-voting/stacking ensemble of all three.
5. **Evaluation:** on the held-out test set only, using Accuracy,
   Precision, Recall, F1, ROC-AUC, PR-AUC, False Positive Rate, False
   Negative Rate, Specificity, Balanced Accuracy, MCC and Cohen's Kappa.
6. **Explainability:** SHAP (global summary, bar, waterfall, dependence,
   force plot) and LIME (four specific cases: correct attack detection, a
   misclassification, a benign flow, an attack flow) on the Random Forest.

## Current Results (real data, 6,000-row stratified sample)

| Model | Accuracy | Precision | Recall | F1 | FPR | ROC-AUC |
|---|---|---|---|---|---|---|
| Random Forest | 0.999 | 0.998 | 1.000 | 0.999 | 0.003 | 1.000 |
| SVM | 0.999 | 0.998 | 1.000 | 0.999 | 0.003 | 1.000 |
| Neural Network | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 |
| Ensemble | 0.999 | 0.998 | 1.000 | 0.999 | 0.003 | 1.000 |

Near-perfect separation is expected and consistent with published
CICIDS2017 benchmarks - DDoS flood traffic is behaviourally very distinct
from normal traffic at the flow-statistics level (packet rate, SYN flag
bursts, timing). Full per-model detail is in
`results/metrics/13_all_model_metrics.csv`; the charts behind this table
(confusion matrices, ROC/PR curves, SHAP, LIME, learning/validation curves,
etc.) are all in `results/figures/`.

## How to Re-run

```bash
pip install -r requirements.txt
jupyter notebook code/01_AI_Intrusion_Detection_Framework.ipynb
```
Run all cells. Outputs regenerate into `models/`, `results/`, and `logs/`
next to wherever you run it from (the notebook auto-detects the project
root via `utils.py`'s location).
