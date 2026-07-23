"""
Shared configuration: file paths and constants used across the pipeline.
Keeping this in one place means every stage (preprocessing, training,
evaluation, explainability) points at the same folders.
"""
import os

# Project root = one level up from src/
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_DIR = os.path.join(ROOT_DIR, "dataset")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")

RAW_CSV = os.path.join(DATASET_DIR, "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")
CLEAN_CSV = os.path.join(DATASET_DIR, "cleaned_features.csv")

RANDOM_STATE = 42

for d in (DATASET_DIR, MODELS_DIR, OUTPUT_DIR, FIGURES_DIR, RESULTS_DIR):
    os.makedirs(d, exist_ok=True)
