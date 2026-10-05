"""Label-free local proxy for the public LB.

Pseudo-label = top third of the rank-averaged 7B VLM scores (Qwen2.5-VL-7B v1+v2, Qwen2-VL-7B), ~1/3 of test is crash.
proxy = AUC of a model's test predictions against that pseudo-label.
Calibrated on 7 past non-VLM submissions: Spearman(proxy, public LB) = 0.99
(0.51 -> 0.462, 0.56 -> 0.595, 0.73 -> 0.609, 0.78 -> 0.669, 0.81 -> 0.710, 0.87 -> 0.746).
It rejects shortcut models reliably but CANNOT certify a model better than the VLMs themselves (ceiling),
and it is circular for anything that contains the 7B VLMs.

    python scripts/proxy_eval.py artifacts/vmae/vmae_test.csv [more files: .csv with clip_id,score or .npy in sample order]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
ids = pd.read_csv(ROOT / "data/sample_submission.csv")["clip_id"]


def vlm(path):
    d = pd.read_csv(ROOT / path)
    d = ids.to_frame().merge(d[d.split == "test"], on="clip_id", how="left")
    w = d.drop(columns=["clip_id", "split"]).values
    return 0.5 * w.max(1) + 0.5 * w.mean(1)


def load(p):
    if p.endswith(".npy"):
        return np.load(p)
    d = pd.read_csv(p)
    col = "score" if "score" in d else "label"
    return ids.to_frame().merge(d, on="clip_id", how="left")[col].values


r = lambda x: rankdata(x) / len(x)
cons = np.mean([r(vlm("artifacts/qwen7b/qwen7b_scores.csv")), r(vlm("artifacts/q7v2/q7v2_scores.csv")),
                r(vlm("artifacts/q2v7b/q2v7b_scores.csv"))], 0)
pseudo = cons > np.quantile(cons, 2 / 3)
b1 = np.load(ROOT / "artifacts/test_blend_B1.npy")
for p in sys.argv[1:]:
    x = load(p)
    assert np.isfinite(x).all() and len(x) == len(ids), p
    print(f"{p}: proxy AUC {roc_auc_score(pseudo, x):.3f} | rank corr vs 7B consensus {spearmanr(x, cons)[0]:.3f} | vs B1 blend {spearmanr(x, b1)[0]:.3f}")
