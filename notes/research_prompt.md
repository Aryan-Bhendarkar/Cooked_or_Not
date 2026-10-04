# Research prompt (paste into Claude chat with Research enabled)

---

You are helping me place near the top of a Kaggle-style computer-vision competition that ends in about 1.5 days. I need concrete, implementable ideas, not a survey. Please research the web (recent papers, Kaggle write-ups, GitHub repos, model cards) and answer the questions at the end.

## 1. The task
- **Input:** a clip = 30 consecutive dashcam frames (JPEG, 384x224, chronological). **Output:** P(a crash occurs inside the clip) per clip. **Metric:** ROC-AUC (rank only).
- **Train:** 3,000 labeled clips = 1,500 pairs. Each pair = one crash clip + one no-crash clip cut from the SAME source video (same `group_id`). The no-crash clip is the EARLIER part of the video: the last 10-15 frames of the no-crash clip equal the first 10-15 frames of the crash clip. So train negatives are "the seconds right before a crash" (tense, often a car already close or braking), not ordinary driving.
- **Test:** 1,750 unlabeled clips, ~1/3 crashes, from different cameras, places, weather and image quality. Test negatives are **ordinary driving from videos with no crash**. Public leaderboard = random 30% of test; private = other 70% decides the rank. Marks = 100*sqrt((S-0.5)/(T-0.5)) with T = mean AUC of the top 10% (I need roughly 0.85+ to score well).
- **Compute:** Kaggle free GPU (T4 x2 or P100, ~25 GPU-hours left, 12 h per session, internet allowed to download weights) and a local RTX 3050 4 GB laptop. About 5 submissions per day.
- **Rules (as I understand them):** pretrained image/video/language models are allowed; no restrictions on techniques; I must NOT hand-label test clips and must NOT try to identify the original public datasets the clips came from (please do not suggest that, and do not help locate the source datasets). If an idea needs external labeled data, flag it separately as "needs a rules check".

## 2. What I measured about the data (EDA)
- Crash clips: mean frame-to-frame pixel difference ramps up over the clip (about 6.8 to 12) and the biggest embedding jump is late in the clip; no-crash clips are flat. Test clips show no such average ramp and the position of the biggest jump is uniform over time, so crash timing in train is biased toward the end of the clip.
- Test frames are much sharper than train (mean edge strength 9.3 vs 6.5); brightness is similar. Test frame differences zigzag with period 2 (looks like frame-rate conversion / duplicated frames). Train frames carry channel/compilation watermarks and badges; test frames carry native camera timestamps. Sharpening or blurring held-out train frames barely changed AUC, so sharpness is not the main cause of the gap.
- A scene-cluster hold-out (8 clusters by embedding) does NOT predict the leaderboard: every cluster still scored 0.88-0.94.

## 3. Everything I tried (CV = 5-fold grouped by group_id; LB = public leaderboard)
| # | Method | CV AUC | Public LB |
|---|---|---|---|
| 1 | DINOv2-base frame embeddings (CLS + mean patch, 30 frames), pooled mean/std/max/diff stats -> logistic regression | 0.943 | **0.595** |
| 2 | Same family, only a 28-number stride-2 embedding-motion curve (shape) -> LR | 0.937 | **0.462** |
| 3 | **CLIP ViT-L/14 zero-shot, no train labels:** frame score = logsumexp(crash prompts) - logsumexp(normal-driving prompts); left and right 224x224 crops of each 2nd frame; clip score = rank average of max, top-3 mean and mean over frames | 0.683 (twin accuracy 86%) | **0.710 (best so far)** |
| 4 | 60 text-prompt scores (max + mean over time, CLIP-L and SigLIP so400m) -> logistic regression ("concept bottleneck"), rank-averaged | 0.848 | 0.669 |
| 5 | Same concept features, pairwise training on (crash minus its twin) differences | 0.757 (twin accuracy 97.7%) | 0.609 |
| - | SigLIP so400m zero-shot alone / blended with CLIP | 0.617 / 0.672 | not submitted |
| - | Self-training simulation on train (pseudo-labels from zero-shot, cross-fitted, labels used only to score) | 0.683 -> 0.722 | not applied to test yet |

Prompt study (CLIP-L, single prompts, AUC on train): best are "a close-up of a car bumper" 0.69, "a blurry shaky image" 0.63, "smoke and fire" 0.61, "a car swerving" 0.59; plain "a car crash" prompts only 0.53-0.59 and "normal driving" prompts are inverted. Prompt wording changed little; a "dashcam" prefix hurt.

**My conclusion so far:** every model fitted on the train labels scored BELOW the zero-shot model that never saw them. The labels teach "tense moment before a crash vs the crash itself" (and source-specific cues), which does not transfer to "crash vs ordinary driving" on new cameras. In-domain CV is useless for model selection here; the public LB is noisy (about +/-0.026).

## 4. What I want you to research and answer
1. **Learning with biased negatives / domain shift:** this looks like positive-unlabeled learning plus covariate and label-shift. Which methods have actually worked when the training negatives are "near-miss" clips and the test negatives are ordinary clips (PU learning, importance weighting, domain-adversarial training, test-time adaptation, self-training/pseudo-labeling, DRO)? Which are realistic in 1.5 days with frozen features?
2. **Better zero-shot / weakly supervised event scores:** video-language or video foundation models that could score "a collision happens in this clip" without fitting to the biased train labels: X-CLIP, InternVideo2, VideoCLIP-XL, LanguageBind, ViCLIP, VideoMAE v2 features, and VLMs (Qwen2.5-VL / Qwen2-VL, InternVL, LLaVA-Video, SmolVLM). For each: does it fit on a T4 (16 GB) for 4,750 clips of 30 small frames, what is the throughput, and what prompting/scoring recipe (yes/no logit, temporal pooling) is known to work for accident detection?
3. **Geometric / physics-based crash cues** that do not depend on appearance: object detection + tracking (box overlap growth, time-to-collision, sudden stops), optical flow or ego-motion anomalies (RAFT, small flow nets), depth change. Which are known to work for dashcam accident detection and how robust are they to frame-rate conversion and sharpness shifts?
4. **Synthetic or re-weighted negatives:** can I build "ordinary driving" negatives from the training videos themselves (e.g. early frames of no-crash clips, stitched or augmented clips) so that a fitted model sees test-like negatives? Any published tricks for crash/anomaly detection with such negatives?
5. **Ensembling and selection under a noisy public LB:** how do strong Kaggle teams choose final submissions when CV is unreliable and the public LB has only ~175 positives? Any specific advice for rank-averaging zero-shot and fitted models with a low weight on the fitted part?
6. **Prior art on dashcam accident detection / anticipation** (methods only; do not identify our data's sources): what strong, simple baselines exist that rely on pretrained features, and what failure modes under cross-camera generalization do papers report?

## 5. Output format I want
A ranked list of at most 8 ideas. For each: (a) one-sentence idea, (b) why it should help given the facts in sections 2-3, (c) expected gain and risk, (d) a concrete recipe (models, hyperparameters, exact scoring formula or prompt text) that I can implement on a Kaggle T4 in under 4 hours, (e) links to the best sources. Put the ideas that need no use of the train labels, or that use them only for light selection, first. End with the three ideas you would run first and why, and any claim you are unsure about.
