# Intrusion Detection Framework — Execution Summary

## Run overview

The completed notebook output records 12,040 CICIDS2017 input rows and 8,026 CSE-CIC-IDS2018 input rows. Cleaning retained 11,882 and 7,924 rows respectively. Twenty-two features were selected through random-forest importance and mutual-information screening. Balanced class weights were selected after comparison with random oversampling, SMOTE and random undersampling.

## Internal evaluation

Logistic Regression, Decision Tree, Naive Bayes, Random Forest, SVM, XGBoost, Neural Network and the ensemble each recorded an F1 score of 1.0000 on the 1,783-row CICIDS2017 smoke test set. Logistic Regression was recorded as the best overall model under the combined F1 and false-positive-rate ranking. The perfect results should be interpreted cautiously because the smoke dataset is designed for pipeline testing and may contain easily separable synthetic patterns.

## External generalisation

Performance declined sharply on the 7,924-row CSE-CIC-IDS2018 smoke set. F1 scores ranged from 0.4875 to 0.5189, while false-positive rates were 0.9998–1.0000. Logistic Regression had the highest recorded external F1 (tied at 0.5189) but a ROC-AUC of only 0.1050. These results show severe cross-dataset distribution shift and indicate that the current models are unsuitable for production deployment without feature harmonisation, representative retraining, threshold calibration and further validation.

## Included evidence

The `figures` directory contains 30 images extracted from the executed notebook, including class distributions, feature analysis, learning curves, model comparisons, confusion matrices, SHAP/LIME explanations and cross-dataset comparisons. The `metrics` directory contains the extracted result tables and the `logs` directory contains the consolidated execution output.
