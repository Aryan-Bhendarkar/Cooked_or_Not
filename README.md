# Cooked or Not — dashcam crash detection

Predict P(crash) for 30-frame dashcam clips. Metric: ROC-AUC. All heavy work (EDA, feature extraction, training) runs as
notebooks on the **Kaggle free GPU**; this machine (RTX 3050, 4 GB) is for smoke tests, small GPU tasks, light analysis and submissions.
Project brief and working rules: [CLAUDE.md](CLAUDE.md).

## Layout
```
CLAUDE.md          project brief + working rules (read first)
notes/             competition brief, concepts, experiment log
kaggle/<step>/     one folder per Kaggle notebook: <step>.ipynb + kernel-metadata.json
src/common.py      folds, pooling, rank averaging, submission writer (helpers used locally)
scripts/           check_submission.py
data/              local copy of competition data (EDA only), gitignored (train/, test/, train_labels.csv, sample_submission.csv, clip_spec.yaml)
artifacts/         downloaded features, folds, OOF/test predictions (gitignored)
submissions/       CSVs to upload (gitignored)
```

## One-time local setup
```powershell
pip install -r requirements.txt
```
Kaggle API token: kaggle.com -> Settings -> API -> Create New Token. Save a `KGAT_...` token to `C:\Users\<you>\.kaggle\access_token`
(or set env var `KAGGLE_API_TOKEN`); a legacy `kaggle.json` also works. Never commit it.
The Kaggle username (`aryanbhendarkar`) is set in every `kaggle/*/kernel-metadata.json`.

## Kaggle notebook setup
No data upload needed: notebooks attach the competition `cooked-or-not` (`competition_sources` in the metadata, or Add Input -> Competitions in the web UI).
1. Verify your phone number (needed for GPU + Internet) and check the GPU quota (~30 h/week) in notebook Settings.
2. Accelerator **GPU T4 x2** (or P100), Internet **On** only when downloading weights, Persistence off.
3. Notebooks locate the data by globbing `/kaggle/input/**/train_labels.csv`.

## Workflow
```powershell
kaggle kernels push -p kaggle/01_extract_features            # or import the .ipynb in the Kaggle web UI
kaggle kernels status <user>/cooked-or-not-01-extract-features
kaggle kernels output <user>/cooked-or-not-01-extract-features -p artifacts/features
python scripts/check_submission.py --submission submissions/<file>.csv
kaggle competitions submit cooked-or-not -f submissions/<file>.csv -m "<message>"
```
Free tier: ~30 GPU-hours/week, 12 h per session, ~20 GB output.
