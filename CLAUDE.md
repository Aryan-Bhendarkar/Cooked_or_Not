# Cooked or Not — Dashcam Crash Detection (college CV competition)

Private Kaggle-style competition run by the college Computer Vision department.
Read `notes/competition.md` for the full brief, `notes/concepts.md` for what Aryan has already been taught,
and `notes/experiment_log.md` for what has been tried.

## How to work with me (Aryan)
- 3rd-year CS student, first competition. You are my **mentor and teammate**: goal is a top rank AND me being able to solve such competitions on my own.
- Be concise, direct, simple language. Give **one best recommendation**, not a menu. Push back honestly when I'm wrong.
- Explain the *why* behind every decision (one or two sentences). Before big steps, tell me what you are about to do and what result would change the plan.
- Occasionally ask me to predict an outcome or explain a result before you reveal it.
- Never silently do large or irreversible things (deleting files, overwriting artifacts, long runs).

## Task in one paragraph
Input: a clip = 30 dashcam frames (`frame_000.jpg`..`frame_029.jpg`, 384x224 JPEG, chronological).
Output: P(crash occurs inside clip) for each of 1,750 test clips. Metric: **ROC-AUC** (rank only; threshold and class ratio irrelevant).
Train: 3,000 clips = 1,500 pairs (a crash clip + a no-crash clip cut from the SAME video, overlapping frames, same `group_id`). Test: different cameras/places/weather, ~1/3 crashes.

## Things that will hurt us (keep in mind in every decision)
1. **Leakage**: twins share frames -> always split by `group_id`. Use `src.common.make_folds()` (cached in `artifacts/folds.csv`) so every experiment uses identical folds.
2. **Domain shift**: grouped CV is still optimistic. Also validate with scene-cluster hold-out and run adversarial validation (train vs test).
3. **Different negatives**: train no-crash = tense seconds BEFORE a crash; test no-crash = ordinary driving. Detect the crash EVENT (sudden motion break), not "dangerous-looking scene".
4. **Shortcuts**: timestamp overlays, black borders, brightness/colour grading, JPEG artefacts identify the source, not the crash. Mask/augment them away.
5. Scene content carries no label signal in train (each scene has 1 pos + 1 neg) -> idea: **pairwise ranking loss within a group**.
6. Crash timing inside the 30 frames: check where it sits in train positives; use random temporal crops. **Never reverse time** as augmentation. Horizontal flip is fine.

## Hard rules (competition rules)
- Do NOT hand-label test clips. Do NOT try to find/identify the original public datasets the clips came from.
- Pretrained image/video backbones are allowed, but **read the Rules page** before using unlabeled test data (pseudo-labels, per-domain normalisation, etc.). If unsure, ask me to check the Rules.
- Public LB is only 30% of test (~525 clips, ~175 crashes): do NOT tune to it. Final rank = private 70%. Pick final submissions by robust CV + sanity checks.

## Grading
Marks = 100 * sqrt((S-0.5)/(T-0.5)), S = my AUC, T = mean AUC of top 10%. The example table in the brief does not match this formula
(except at 0.85/0.90) — trust the formula, and ask the instructor which applies. Practical goal: be in the top ~10%.

## Timeline
Competition started ~Sep 21 2026; on Sep 29 it showed "6 days to go" -> ends **Oct 5 2026 18:29:59 UTC (23:59 IST)**.

## Environments (two-machine workflow)
- **Local (Windows, VS Code + Claude Code, RTX 3050 Laptop 4 GB, torch 2.6+cu124 already installed)**: editing notebooks, submissions, and **small tasks on the local GPU** (smoke tests of a notebook on ~50 clips, small-model inference, tiny heads on saved features). Use `device="cuda"` locally when it fits in 4 GB (batch small, fp16); don't use it for full-dataset extraction or fine-tuning.
- **Kaggle free GPU (T4 x2 / P100, ~30 GPU-hours/week, 12h max session)**: full-scale runs: embeddings over all 142k frames, optical flow, fine-tuning, final training. Debug locally first so Kaggle quota isn't wasted on bugs.
- Data: `data/` (gitignored; same layout as the competition: `train/<clip_id>/frame_XXX.jpg`, `test/...`, `train_labels.csv`, `sample_submission.csv`, `clip_spec.yaml`).
  On Kaggle the data is attached directly as the competition source `cooked-or-not` (no upload needed; `/kaggle/input/...`). Notebooks find the data root by globbing for `train_labels.csv`.
- Kaggle -> local handoff: Kaggle notebooks write to `/kaggle/working/features/*.npy`; download with `kaggle kernels output <user>/<slug> -p artifacts/features` and continue locally.

## Code conventions
- Notebooks for anything that runs on Kaggle (`kaggle/<step>/`, each with `kernel-metadata.json`); keep them **thin and self-contained**. Reusable logic lives in `src/common.py` (copy small helpers into Kaggle notebooks if needed).
- Seed 42 everywhere. Folds: 5-fold StratifiedGroupKFold by `group_id`.
- Every experiment saves `artifacts/oof_<name>.npy` (OOF preds in train_labels.csv order) and `artifacts/test_<name>.npy` (test preds in sample_submission.csv order).
- Blend by **rank average** (`src.common.rank_average`), never by raw probability averaging.
- Submissions go to `submissions/<YYYYMMDD>_<name>_cv<auc>.csv`; validate with `scripts/check_submission.py` (copy it from the competition repo into `scripts/`).
- After each experiment append a row to `notes/experiment_log.md` (idea, CV AUC, cluster-holdout AUC, public LB, takeaway).
- Don't commit `data/`, `artifacts/` or `submissions/` contents.

## Roadmap / status  (update as we go)
- [x] Concepts taught (see notes/concepts.md); self-check answered
- [x] Clean repo setup (data flattened into `data/`, EDA moved to `kaggle/00_eda`, `.vscode/`, check_submission.py)
- [x] Kaggle configured (API token in ~/.kaggle/access_token, username set, notebooks attach competition `cooked-or-not`)
- [x] Run `kaggle/00_eda`, record findings in notes/competition.md ("EDA findings")
- [ ] Extract DINOv2 frame embeddings on Kaggle (`kaggle/01_extract_features`)
- [ ] Baseline: pooled embeddings + temporal diffs -> logistic regression, grouped CV
- [ ] Motion features (frame diff, optical flow stats) -> LightGBM
- [ ] Scene-cluster hold-out validation + adversarial validation
- [ ] Fine-tune video/temporal model with pairwise ranking loss (Kaggle GPU)
- [ ] Rank-average ensemble, choose 2 final submissions
