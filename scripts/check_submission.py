"""Validate a submission CSV against sample_submission.csv.

Usage:
    python scripts/check_submission.py --submission submissions/x.csv [--sample data/sample_submission.csv]
"""
import argparse
import sys

import numpy as np
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--submission", required=True)
p.add_argument("--sample", default="data/sample_submission.csv")
a = p.parse_args()

sub, ref = pd.read_csv(a.submission), pd.read_csv(a.sample)
errors = []
if list(sub.columns) != ["clip_id", "label"]:
    errors.append(f"columns must be exactly ['clip_id', 'label'], got {list(sub.columns)}")
else:
    if len(sub) != len(ref):
        errors.append(f"expected {len(ref)} rows, got {len(sub)}")
    if sub["clip_id"].duplicated().any():
        errors.append("duplicate clip_id")
    if set(sub["clip_id"]) != set(ref["clip_id"]):
        errors.append("clip_id set differs from sample_submission")
    lab = pd.to_numeric(sub["label"], errors="coerce")
    if lab.isna().any() or not np.isfinite(lab).all():
        errors.append("label has empty / non-numeric / non-finite values")
    elif lab.min() < 0 or lab.max() > 1:
        errors.append("label must be in [0, 1]")
    elif lab.nunique() == 1:
        errors.append("warning: all predictions identical (AUC = 0.5)")

if errors:
    print("INVALID:\n- " + "\n- ".join(errors))
    sys.exit(1)
print(f"OK: {len(sub)} rows, label range [{sub.label.min():.4f}, {sub.label.max():.4f}]")
