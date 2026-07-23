"""
Stage 4 — Explainable AI (SHAP & LIME)
Loads the trained Random Forest model and generates a SHAP global
feature-importance chart and a LIME local explanation for one test
instance. Both are saved to output/figures/.

Run:  python src/explainability.py
"""
import pickle
import joblib
import numpy as np
import matplotlib.pyplot as plt

import shap
from lime.lime_tabular import LimeTabularExplainer

from config import MODELS_DIR, FIGURES_DIR


def main():
    rf_model = joblib.load(f"{MODELS_DIR}/random_forest.pkl")
    feature_names = joblib.load(f"{MODELS_DIR}/feature_names.pkl")
    X_test_s = np.load(f"{MODELS_DIR}/X_test_s.npy")
    X_train_bal = np.load(f"{MODELS_DIR}/X_train_bal.npy")

    print("Generating SHAP Feature Importance...")
    explainer = shap.TreeExplainer(rf_model)
    sample_subset = X_test_s[:100]
    shap_values = explainer.shap_values(sample_subset)
    sv = shap_values[1] if isinstance(shap_values, list) else shap_values

    shap.summary_plot(sv, sample_subset, feature_names=feature_names,
                       plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/shap_feature_importance.png", dpi=150)
    plt.close()

    print("Generating LIME Explanation...")
    lime_explainer = LimeTabularExplainer(
        X_train_bal,
        feature_names=feature_names,
        class_names=["BENIGN", "DDoS"],
        discretize_continuous=True
    )

    lime_exp = lime_explainer.explain_instance(
        X_test_s[0], rf_model.predict_proba, num_features=6
    )

    fig = lime_exp.as_pyplot_figure()
    plt.tight_layout()
    fig.savefig(f"{FIGURES_DIR}/lime_explanation.png", dpi=150)
    plt.close()

    print(f"SHAP and LIME figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
