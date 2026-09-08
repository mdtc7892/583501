# AI-Driven Intrusion Detection Framework

This repository contains the technical artefact for designing and evaluating an AI-driven intrusion-detection framework for real-time banking network security.

## Project structure

```text
AI_Intrusion_Detection_Framework/
├── code/
│   ├── 01_AI_Intrusion_Detection_Framework.ipynb
│   └── utils.py
├── dataset/
│   ├── raw/
│   └── processed/
├── logs/
├── models/
├── results/
│   ├── figures/
│   ├── metrics/
│   └── reports/
├── README.md
└── requirements.txt
```

## Dataset setup

Place the source CSV files in `dataset/raw/` using these names:

- `cicids2017_smoke.csv` for the primary CICIDS2017 data.
- `cic2018_smoke.csv` for optional CSE-CIC-IDS2018 external testing.

The full datasets may also be used after updating the two paths in the notebook configuration cell. Dataset files are intentionally not included because of their size and licensing/distribution conditions.

## Run

1. Create a Python 3.10 or 3.11 virtual environment.
2. Install dependencies with `pip install -r requirements.txt`.
3. Start Jupyter from the project root: `jupyter lab`.
4. Open `code/01_AI_Intrusion_Detection_Framework.ipynb` and run the cells in order.

Generated files are written automatically to `logs/`, `models/`, and the relevant `results/` subdirectories.

## Multi-model architecture

The framework deliberately evaluates several complementary model families rather than relying on a single classifier. This supports a balanced comparison of predictive performance, interpretability, computational cost and generalisation.

| Model | Description | Main value in intrusion detection |
|---|---|---|
| Logistic Regression | A regularised linear classifier that estimates the probability of an attack from a weighted combination of network-flow features. | Transparent baseline, fast training and straightforward audit explanations. |
| Decision Tree | A rule-based model that recursively splits observations into increasingly homogeneous classes. | Highly interpretable if–then rules that security analysts can inspect. |
| Gaussian Naive Bayes | A probabilistic classifier that estimates class likelihoods under a conditional-independence assumption. | Very fast benchmark and useful for identifying whether simple distributional patterns separate traffic classes. |
| Random Forest | An ensemble of decorrelated decision trees trained on bootstrapped samples and feature subsets. | Captures non-linear interactions and usually provides robust feature-importance estimates. |
| Support Vector Machine (SVM) | A maximum-margin classifier that separates classes in a transformed feature space; the notebook applies a scalability cap for large training sets. | Effective in high-dimensional feature spaces, although more expensive to tune and less directly interpretable. |
| XGBoost | A gradient-boosted tree model that sequentially corrects errors made by earlier trees, with regularisation and class-imbalance weighting. | Strong non-linear modelling and competitive performance on tabular flow data. |
| Dense Neural Network | A feed-forward network using dense layers, batch normalisation, dropout, early stopping and learning-rate scheduling. | Learns complex feature combinations and provides a flexible deep-learning comparison. |
| Soft-voting ensemble | A probability-level combination of Random Forest, SVM, XGBoost and the neural network. | Reduces dependence on one inductive bias and provides a combined decision score. |

### Selection and evaluation logic

Feature selection combines variance filtering, correlation filtering, random-forest importance, mutual information and recursive feature elimination. Each model is trained using leakage-safe pipelines, stratified cross-validation and hyperparameter search. The models are compared using accuracy, precision, recall, F1 score, ROC-AUC, PR-AUC and false-positive rate. False-positive rate is treated as a key banking-security measure because excessive alerts can overwhelm analysts and reduce trust in the detection system.

The neural network and ensemble complement the interpretable classical baselines: the baselines make decisions easier to audit, while the non-linear models can capture complex attack signatures. SHAP and LIME outputs in `results/figures/` provide post-hoc explanations for the tree-based and major model predictions. Internal CICIDS2017 results and external CSE-CIC-IDS2018 generalisation results must be read together; strong internal smoke-test performance does not by itself demonstrate production readiness.
