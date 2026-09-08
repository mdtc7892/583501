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

Multi-Model Intrusion Detection Architecture

The proposed framework adopts a multi-model machine-learning architecture to evaluate intrusion-detection performance from complementary perspectives. Rather than depending on a single algorithm, the framework compares interpretable statistical models, rule-based classifiers, ensemble learners, support-vector methods and neural networks. This approach enables a more robust assessment of predictive accuracy, computational efficiency, interpretability and cross-dataset generalisation.

Model	Professional description	Role in the framework
Logistic Regression	A regularised linear classification algorithm that estimates the probability of benign or malicious network activity from a weighted combination of selected features.	Serves as an interpretable and computationally efficient baseline. Its coefficients can support audit trails and explainable security decisions.
Decision Tree	A hierarchical rule-based classifier that recursively partitions observations according to feature values until a class decision is reached.	Provides transparent if–then rules that can be reviewed by security analysts and translated into operational detection logic.
Gaussian Naive Bayes	A probabilistic classifier that applies Bayes’ theorem while assuming conditional independence between features and approximately Gaussian feature distributions.	Provides a fast probabilistic benchmark and tests whether relatively simple statistical relationships can distinguish normal and malicious traffic.
Random Forest	An ensemble of decision trees trained on bootstrapped data samples and randomly selected feature subsets.	Captures non-linear relationships, reduces the instability of a single tree and provides feature-importance information.
Support Vector Machine	A maximum-margin classifier that identifies an optimal decision boundary between traffic classes, with kernel-based transformations available for complex patterns.	Supports high-dimensional classification and is useful where attack and benign traffic require a well-defined separating boundary.
XGBoost	A gradient-boosting algorithm that sequentially develops decision trees, with each new tree correcting errors made by previous trees.	Models complex non-linear interactions and offers strong performance on structured network-flow data, with regularisation to reduce overfitting.
Dense Neural Network	A feed-forward neural architecture consisting of fully connected layers, batch normalisation, dropout and early-stopping controls.	Learns complex feature interactions that may not be captured by conventional statistical or tree-based algorithms.
Soft-Voting Ensemble	A probability-level ensemble that combines predictions from Random Forest, SVM, XGBoost and the neural network.	Integrates different modelling assumptions and produces a consolidated intrusion-risk score.
Model development and evaluation

The modelling pipeline applies variance-threshold filtering, correlation analysis, random-forest importance, mutual information and recursive feature elimination to identify relevant predictors. Each classifier is implemented using leakage-safe preprocessing pipelines and evaluated through stratified cross-validation and hyperparameter optimisation.

Performance is assessed using accuracy, precision, recall, F1-score, ROC-AUC, PR-AUC and false-positive rate. The false-positive rate is particularly important in banking environments because excessive false alerts can increase analyst workload, delay the investigation of genuine incidents and reduce confidence in automated security controls.

The models also serve different operational purposes. Logistic Regression and Decision Trees offer stronger interpretability, whereas Random Forest, XGBoost and the Neural Network are better suited to modelling complex non-linear attack behaviour. The SVM provides an additional high-dimensional decision-boundary perspective. SHAP and LIME are used to improve the explainability of complex predictions, while the soft-voting ensemble combines the probability outputs of several high-performing models.

Final model selection considers both predictive performance and operational suitability. Internal CICIDS2017 results are compared with external CSE-CIC-IDS2018 results to identify distribution shift and assess generalisation. Consequently, a model achieving excellent internal performance is not automatically considered production-ready unless it also demonstrates stable performance on representative external data, acceptable false-positive rates and explainable decisions.
