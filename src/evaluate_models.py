"""
Stage 3 — Model Evaluation & Metrics
Loads the trained models from models/, evaluates each on the held-out
test set, saves per-model confusion matrices and a comparison chart to
output/figures/, and writes results/model_comparison.csv.

Run:  python src/evaluate_models.py
"""
import pickle
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix,
                              ConfusionMatrixDisplay)

from config import MODELS_DIR, FIGURES_DIR, RESULTS_DIR


def calculate_fpr(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return fp / (fp + tn) if (fp + tn) > 0 else 0.0


def load_artifacts():
    X_test_s = np.load(f"{MODELS_DIR}/X_test_s.npy")
    with open(f"{MODELS_DIR}/y_test.pkl", "rb") as f:
        y_test = pickle.load(f)

    model_files = {
        "Random Forest": "random_forest.pkl",
        "SVM": "svm.pkl",
        "Neural Network (MLP)": "neural_network_mlp.pkl",
    }
    trained_models = {name: joblib.load(f"{MODELS_DIR}/{fname}")
                       for name, fname in model_files.items()}
    return trained_models, X_test_s, y_test


def main():
    trained_models, X_test_s, y_test = load_artifacts()

    evaluation_results = []
    fig, axes = plt.subplots(1, len(trained_models), figsize=(15, 4))

    for i, (name, model) in enumerate(trained_models.items()):
        y_pred = model.predict(X_test_s)
        y_prob = model.predict_proba(X_test_s)[:, 1]

        evaluation_results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1-Score": f1_score(y_test, y_pred),
            "False Positive Rate": calculate_fpr(y_test, y_pred),
            "ROC-AUC": roc_auc_score(y_test, y_prob),
        })

        ConfusionMatrixDisplay.from_predictions(
            y_test, y_pred, display_labels=["BENIGN", "DDoS"],
            ax=axes[i], cmap="Blues", colorbar=False
        )
        axes[i].set_title(name)

    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/confusion_matrices.png", dpi=150)
    plt.close()

    results_df = pd.DataFrame(evaluation_results).round(4)
    print(results_df)

    results_df.to_csv(f"{RESULTS_DIR}/model_comparison.csv", index=False)
    print(f"Results saved to: {RESULTS_DIR}/model_comparison.csv")
    print(f"Confusion matrices saved to: {FIGURES_DIR}/confusion_matrices.png")

    return results_df


if __name__ == "__main__":
    main()
