"""
Runs the full pipeline end-to-end, in order:
  1. data_preprocessing.py  -> dataset/cleaned_features.csv
  2. train_models.py        -> models/*.pkl
  3. evaluate_models.py     -> results/model_comparison.csv, output/figures/confusion_matrices.png
  4. explainability.py      -> output/figures/shap_feature_importance.png, lime_explanation.png

Run:  python src/run_all.py
"""
import data_preprocessing
import train_models
import evaluate_models
import explainability

if __name__ == "__main__":
    print("\n=== STAGE 1: Data Preprocessing ===")
    data_preprocessing.main()

    print("\n=== STAGE 2: Train/Test Split, SMOTE, Model Training ===")
    train_models.main()

    print("\n=== STAGE 3: Model Evaluation ===")
    evaluate_models.main()

    print("\n=== STAGE 4: Explainable AI (SHAP & LIME) ===")
    explainability.main()

    print("\nPipeline complete. Check results/ and output/figures/.")
