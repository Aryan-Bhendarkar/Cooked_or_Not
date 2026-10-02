# Learning protocol — how Claude Code and I work together

Goal: finish this competition strong AND become someone who can solve the next one without help.
Rule of thumb: **I own the hypothesis, Claude writes the plumbing, and I must be able to explain every piece.**

## For every task
1. **Plan first (3 lines):** what we're doing, why, and what result we expect. For any real experiment, ask me for my one-line **prediction** ("this will raise CV AUC because...") and record it in `notes/experiment_log.md`.
2. **Small steps:** write code in chunks of roughly 60 lines or fewer, with comments that say *why*, not what. After each chunk, explain it in at most 5 plain sentences.
3. **After the result:** compare it with my prediction, ask me ONE "why do you think this happened?" question, wait for my answer, then give yours. Surprises are where the learning is.
4. **Build-it-yourself:** at each milestone pick one component and have ME write it before showing yours (GroupKFold split, AUC from scratch, temporal pooling, pairwise ranking loss, frame-difference/flow stats). Review my version.
5. **Quiz:** 3 short questions per milestone. Log what I got wrong in `notes/concepts.md` under "Gaps".
6. **Say what shift a trick targets:** any augmentation/normalisation/feature must be justified by a specific train->test difference (camera, overlay, brightness, motion, class ratio).

## Sanity checks before trusting ANY score (non-negotiable)
- **Label-shuffle test:** shuffle train labels, rerun the same pipeline -> CV AUC must be about 0.50. If not, there is leakage.
- **Group leak check:** assert no `group_id` appears in both train and validation of any fold.
- **Tiny-batch overfit:** a neural model must be able to overfit 32 clips to ~100% train accuracy; if not, the code is broken.
- **Order check:** predictions are in `train_labels.csv` order (OOF) / `sample_submission.csv` order (test). Assert ids match before writing.
- **Shapes/dtypes/NaNs asserted** at every stage boundary.
- **Determinism:** same seed + same folds -> same score.

## Habits (from Kaggle Grandmaster advice)
- Robust local validation first; then iterate against it. Simulate a "public/private split" inside CV so I don't get fooled by a noisy leaderboard.
- Always analyse train vs test differences before modelling.
- MVP baseline first, then one change at a time. Models are hypotheses.
- Track every experiment; save OOF and test predictions of everything (they become ensemble ingredients).
- After the competition: read the top solutions, then rebuild the key idea from scratch.
