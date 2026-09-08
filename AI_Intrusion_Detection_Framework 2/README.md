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
