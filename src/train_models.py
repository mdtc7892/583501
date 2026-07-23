"""
Stage 2 — Train/Test Split, SMOTE Balancing, and Model Training
Trains Random Forest, SVM, and MLP (Neural Network) classifiers on the
cleaned CICIDS2017 DDoS features and saves the fitted models + scaler
to models/.

Note on SVM: an RBF-kernel SVM scales roughly O(n^2)-O(n^3) with the
number of training rows. On the full ~170k-row balanced training set
this is not practical to fit in a normal session, so SVM is fit on a
stratified subsample of the (already SMOTE-balanced) training data
(see SVM_TRAIN_CAP below). Random Forest and the MLP are trained on
the full balanced training set. All three models are evaluated on the
SAME full, untouched test set, so reported metrics remain a fair,
full-dataset comparison. This is a standard, documented practice when
benchmarking kernel SVMs on large network-flow datasets.

Run:  python src/train_models.py
"""
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE

from config import CLEAN_CSV, MODELS_DIR, RANDOM_STATE

np.random.seed(RANDOM_STATE)

SVM_TRAIN_CAP = 20000  # stratified subsample size used only for SVM fitting


def load_clean_data():
    df = pd.read_csv(CLEAN_CSV)
    y = df["Label"]
    X = df.drop(columns=["Label"])
    return X, y


def split_scale_balance(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    print("Class distribution before SMOTE:", np.bincount(y_train))

    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_s, y_train)

    print("Class distribution after SMOTE :", np.bincount(y_train_bal))

    return X_train_bal, X_test_s, y_train_bal, y_test, scaler


def stratified_subsample(X, y, cap, random_state=RANDOM_STATE):
    if len(y) <= cap:
        return X, y
    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]
    rng = np.random.RandomState(random_state)
    n_pos = int(cap * len(idx_pos) / len(y))
    n_neg = cap - n_pos
    sel_pos = rng.choice(idx_pos, size=min(n_pos, len(idx_pos)), replace=False)
    sel_neg = rng.choice(idx_neg, size=min(n_neg, len(idx_neg)), replace=False)
    sel = np.concatenate([sel_pos, sel_neg])
    rng.shuffle(sel)
    return X[sel], y[sel]


def train_all(X_train_bal, y_train_bal):
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "SVM": SVC(kernel="rbf", C=1.0, probability=True, random_state=RANDOM_STATE),
        "Neural Network (MLP)": MLPClassifier(
            hidden_layer_sizes=(64, 32), max_iter=200, random_state=RANDOM_STATE
        ),
    }

    trained_models = {}
    for name, model in models.items():
        print(f"Training {name}...")
        if name == "SVM":
            X_fit, y_fit = stratified_subsample(X_train_bal, y_train_bal, SVM_TRAIN_CAP)
            print(f"  (SVM fit on stratified subsample: {X_fit.shape[0]} rows)")
            model.fit(X_fit, y_fit)
        else:
            model.fit(X_train_bal, y_train_bal)
        trained_models[name] = model

    print("All models trained successfully!")
    return trained_models


def main():
    X, y = load_clean_data()
    X_train_bal, X_test_s, y_train_bal, y_test, scaler = split_scale_balance(X, y)
    trained_models = train_all(X_train_bal, y_train_bal)

    joblib.dump(scaler, f"{MODELS_DIR}/scaler.pkl")
    for name, model in trained_models.items():
        fname = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        joblib.dump(model, f"{MODELS_DIR}/{fname}.pkl")

    np.save(f"{MODELS_DIR}/X_test_s.npy", X_test_s)
    y_test.to_numpy().dump(f"{MODELS_DIR}/y_test.pkl")
    np.save(f"{MODELS_DIR}/X_train_bal.npy", X_train_bal)
    joblib.dump(list(X.columns), f"{MODELS_DIR}/feature_names.pkl")

    print(f"Models, scaler, and split arrays saved to: {MODELS_DIR}")


if __name__ == "__main__":
    main()
