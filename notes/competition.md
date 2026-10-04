# Competition brief (condensed from the official page)

**Goal**: for every test clip output a crash probability in [0,1]. Ranked by ROC-AUC.

## Data
- `train/<clip_id>/frame_000.jpg … frame_029.jpg` — 3,000 labeled clips (1,500 crash / 1,500 no-crash, exactly balanced)
- `test/<clip_id>/…` — 1,750 unlabeled clips
- `train_labels.csv`: `clip_id,label,group_id` (group_id only in train)
- `sample_submission.csv`: `clip_id,label` (all 0.5)
- `clip_spec.yaml`: 30 frames, 384x224 JPEG, same for train and test
- Clip IDs random (`c123456`), CSV rows shuffled -> IDs carry no information.
- Front-facing dashcam views. 30 consecutive frames, chronological.

## Pairs / groups
Training clips come in pairs from the same source video: one no-crash clip and one crash clip whose frames partly overlap in time; both share `group_id`.
Must use grouped splits (GroupKFold / StratifiedGroupKFold).

## Intended train -> test shifts
- Cameras, places, regions, lighting, weather differ.
- Crash share: train 50%, test ~1/3 (metric is rank-based so threshold/ratio don't matter).
- Train no-crash = seconds before a crash in videos that later crash. Test no-crash = ordinary driving, no crash in the video.
- Image quality: resized, black borders cropped, re-encoded; compression differences and overlays (date/time stamp) may remain.

## Evaluation
ROC-AUC between predicted probability and label. Public LB = random 30% of test (visible), private LB = other 70% (revealed at end; decides final rank). Final rank uses the private score of selected Final Submissions.

## Submission format
Header + exactly columns `clip_id,label`; one row per clip in sample_submission; label numeric in [0,1]; no duplicates/empties.
Validate: `python scripts/check_submission.py --submission submissions/<file>.csv --sample data/sample_submission.csv`

## Grading
Marks = 100 * sqrt((S - 0.5) / (T - 0.5)) capped to [0,100]; S = my ROC-AUC, T = average ROC-AUC of top 10% participants.
Example table in the brief (T=0.90) disagrees with the formula: formula gives 0.60->50, 0.70->71, 0.80->87 (table: 58, 75, 89). Ask instructor.

## Rules of thumb from the brief
- Pretrained image/video backbones allowed (see Rules page for exact limits — TODO: paste here).
- Labeling test by hand or trying to find the original datasets is NOT allowed.
- Validate by group_id; prefer methods robust to camera/scene change.

## Rules page
- (Oct 2, per Aryan) No restrictions on techniques; data was downloaded from Kaggle itself, so notebooks attach the competition data directly (no upload).
- Still unknown: max submissions/day, number of final submissions. Hand-labeling test and finding source datasets remain banned.
- (Oct 4, per Aryan, after reading the Rules page) Using unlabeled test clips (self-training / pseudo-labels / test-time adaptation) is ALLOWED. Hand-labeling test and identifying source datasets remain banned.
- Grading formula confirmed from the course doc: Marks = 100*sqrt((S-0.5)/(T-0.5)), capped to [0,100].

## EDA findings (fill after running kaggle/00_eda)
(Oct 2, from kaggle/00_eda, 200 clips per group; plots in artifacts/eda/eda/)
- Crash timing in positives: no single spike. Mean |frame diff| of crash clips RAMPS from ~6.8 (t=0) to ~12 (t=20-28); no-crash stays flat ~7.4. Crash happens in the later half, build-up is gradual.
- Test vs train visual differences: train frames carry channel/compilation watermarks and "21+" badges (same in both clips of a pair, so no label signal but a source shortcut); test has native camera date/time stamps, no watermarks. Different camera styles/regions, test looks sharper.
- Twin overlap pattern: the crash clip's first 10-15 frames (0-9/0-14) equal the no-crash twin's LAST frames (20-29/19-29). So the no-crash clip is the EARLIER one; crash clip starts inside it and runs on into the crash. Confirms leakage risk with random folds.
- Brightness / sharpness / motion shift: brightness same (108.8 vs 109.5), contrast similar (61.8 vs 64.5), **sharpness differs strongly (train 6.5, test 9.3)**. Mean frame-diff: train crash 9.6, no-crash 7.5, test 10.4 -> raw motion magnitude is inflated by sharpness, not trustworthy across domains.
- **Test frame-diff curve zigzags with period 2** (alternating ~9.2 / ~11): test frames have a duplicate/interpolation pattern from frame-rate conversion. Train shows only weak bumps every ~8 frames. => use stride-2 differences or smoothing, and scale-free features (late/early motion ratio), not raw magnitudes.
- Plan changes: (1) blur/downscale augmentation or fixed pre-blur so sharpness is not a domain cue, (2) relative temporal features instead of absolute motion, (3) crop mostly later frames for positives, (4) mask watermark/timestamp regions.
