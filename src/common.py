"""Shared helpers. Import from scripts/notebooks with:

    import sys; sys.path.append("..")   # if running from a subfolder
    from src.common import *
"""
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ART = ROOT / "artifacts"
SUBS = ROOT / "submissions"
SEED = 42
N_FRAMES = 30


# ------------------------------------------------------------------ data
def load_labels(data=DATA) -> pd.DataFrame:
    return pd.read_csv(Path(data) / "train_labels.csv")


def load_test_ids(data=DATA) -> list:
    return pd.read_csv(Path(data) / "sample_submission.csv")["clip_id"].tolist()


def load_clip(split: str, clip_id: str, data=DATA) -> np.ndarray:
    """uint8 array (30, 224, 384, 3), chronological."""
    d = Path(data) / split / clip_id
    return np.stack([np.asarray(Image.open(d / f"frame_{i:03d}.jpg").convert("RGB")) for i in range(N_FRAMES)])


# ------------------------------------------------------------------ validation
def make_folds(n_splits: int = 5, seed: int = SEED, data=DATA) -> pd.DataFrame:
    """Grouped, stratified folds. Cached in artifacts/folds.csv so every experiment uses the SAME folds."""
    path = ART / "folds.csv"
    if path.exists():
        return pd.read_csv(path)
    lab = load_labels(data)
    lab["fold"] = -1
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for f, (_, va) in enumerate(sgkf.split(lab, lab["label"], groups=lab["group_id"])):
        lab.loc[lab.index[va], "fold"] = f
    ART.mkdir(exist_ok=True)
    lab[["clip_id", "fold"]].to_csv(path, index=False)
    return lab[["clip_id", "fold"]]


# ------------------------------------------------------------------ features
def pool_features(emb: np.ndarray) -> np.ndarray:
    """emb: (N, 30, D) per-frame embeddings -> (N, 5*D) clip features.

    mean / max / std over time + mean and max of |consecutive-frame difference|
    (large jumps in embedding space = abrupt scene change = possible crash).
    """
    emb = emb.astype(np.float32)
    d = np.abs(emb[:, 1:] - emb[:, :-1])
    return np.concatenate(
        [emb.mean(1), emb.max(1), emb.std(1), d.mean(1), d.max(1)], axis=1
    )


# ------------------------------------------------------------------ ensembling / output
def rank01(x) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return (rankdata(x) - 1) / (len(x) - 1)


def rank_average(preds, weights=None) -> np.ndarray:
    """Average predictions by RANK (AUC only cares about order)."""
    r = np.stack([rank01(p) for p in preds])
    return np.average(r, axis=0, weights=weights)


def write_submission(test_pred, name: str, cv: float, data=DATA) -> Path:
    """Write submissions/<date>_<name>_cv<auc>.csv in the sample_submission row order."""
    from datetime import date

    ids = load_test_ids(data)
    test_pred = np.asarray(test_pred, dtype=np.float64)
    assert len(ids) == len(test_pred), "prediction length != sample_submission rows"
    assert np.isfinite(test_pred).all()
    p = rank01(test_pred)  # in [0,1]
    SUBS.mkdir(exist_ok=True)
    out = SUBS / f"{date.today():%Y%m%d}_{name}_cv{cv:.4f}.csv"
    pd.DataFrame({"clip_id": ids, "label": p}).to_csv(out, index=False)
    return out
