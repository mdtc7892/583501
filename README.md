# DDoS Attack Detection & Intelligent Intrusion Detection System (IDS)

An advanced, machine learning-driven Cybersecurity Framework designed for real-time detection, analysis, and classification of Distributed Denial of Service (DDoS) and network intrusion attacks.

---

## 📌 Project Overview

Modern network infrastructures face continuous threats from sophisticated DDoS attacks that disrupt critical services. This project implements a comprehensive **Intrusion Detection System (IDS)** powered by Machine Learning and Deep Learning architectures. 

The framework processes high-dimensional network flow statistics, performs automated feature engineering and quality checks, trains multiple high-accuracy classifier models, and provides actionable insights and analytical figures.

---

## 🛠️ Key Features

- **Multi-Model Architecture**: Includes Logistic Regression, Random Forest, XGBoost, and Deep Neural Networks (Keras/TensorFlow).
- **Automated Data Quality & Preprocessing**: Cleaning, scaling, outlier removal, and class balancing pipelines.
- **Robust Feature Selection**: Identifies critical network traffic indicators (e.g., Flow Duration, Packet Length Statistics, Flag Counts).
- **Comprehensive Evaluation**: Automated generation of Confusion Matrices, ROC-AUC curves, Precision-Recall metrics, and Classification Reports.
- **Production-Ready Artifacts**: Saved trained models (`.joblib`, `.keras`) and structured JSON summaries for easy deployment and integration.

---

## 📁 Repository Structure

```text
.
├── code/
│   ├── data_preprocessing.py     # Data cleaning, scaling, and feature pipeline
│   ├── train_models.py           # Training scripts for ML & Deep Learning models
│   ├── evaluate.py               # Evaluation metrics and figure generator
│   └── utils.py                  # Helper functions and logger setup
├── models/                       # Directory containing pre-trained model artifacts
│   ├── xgboost_ids.joblib
│   ├── random_forest_ids.joblib
│   └── deep_learning_ids.keras
├── results/                      # Output directory for evaluations
│   ├── figures/                  # Visualization plots (ROC, Confusion Matrix, Feature Importance)
│   └── reports/                  # Detailed evaluation reports in JSON format
├── requirements.txt              # Environment dependencies
└── README.md                     # Documentation and Setup Guide
