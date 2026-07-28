# Dataset

Both files here are the **real CICIDS2017** DDoS-vs-BENIGN capture (not
synthetic) - Friday afternoon DDoS attack window from the Canadian Institute
for Cybersecurity's benchmark IDS dataset.

- `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv` - the full file,
  225,745 flows (97,718 BENIGN / 128,027 DDoS), obtained from a public
  GitHub mirror: https://github.com/StarterArcher/CICIDS2017
- `cicids2017_real_sample.csv` - a stratified 6,000-row sample of the file
  above (same real class ratio), used as the notebook's default
  `CONFIG.data_raw_dir` so the full pipeline - EDA, feature engineering,
  hyperparameter search, ensembling, SHAP/LIME - finishes in a practical
  amount of time.

**To train on the full 225,745-row file instead:** open
`code/01_AI_Intrusion_Detection_Framework.ipynb`, find the `CONFIG =
utils.ProjectConfig(...)` cell, and change:
```python
data_raw_dir=Path("data/raw/cicids2017_real_sample.csv"),
```
to
```python
data_raw_dir=Path("data/raw/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"),
```
(and put the CSV in `data/raw/` relative to the notebook, or adjust the
path). Expect SVM/hyperparameter-search/SHAP steps to take much longer at
that size - see the "MAX_SVM_TRAIN" cap and subsampling notes inside the
notebook.

## Citation

If this data is used or redistributed further, cite:
Iman Sharafaldin, Arash Habibi Lashkari, and Ali A. Ghorbani, "Toward
Generating a New Intrusion Detection Dataset and Intrusion Traffic
Characterization", ICISSP 2018.

## Second dataset mentioned in the report (CSE-CIC-IDS2018)

For the external-validation / generalisation step mentioned in the
Methodology chapter, not included here (multi-GB):
- Official (AWS Open Data): https://registry.opendata.aws/cse-cic-ids2018/
- Kaggle mirror: https://www.kaggle.com/datasets/solarmainframe/ids-intrusion-csv
