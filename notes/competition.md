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
- Grading formula confirmed from the course doc: Marks = 100*sqrt((S-0.5)/(T-0.5)), capped to [0,100].

## EDA findings (fill after running kaggle/00_eda)
- Crash timing in positives:
- Test vs train visual differences (overlays, night, resolution feel):
- Twin overlap pattern (which frames shared):
- Brightness / sharpness / motion shift:
