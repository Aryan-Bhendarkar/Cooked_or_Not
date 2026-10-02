# Concepts taught so far (one-page recap)

1. **Clip = array** (30,224,384,3). Normalise with the backbone's mean/std. A single frame can't show a crash; change over time does.
2. **Embeddings / transfer learning**: pretrained net (DINOv2, CLIP, ResNet) -> vector per image. Frozen features = fast and safe with 3k clips; fine-tuning = stronger but overfits.
3. **Crash signal**: sudden motion change, camera jolt, objects merging/occluding, rapid expansion. Tools: frame differencing, optical flow (magnitude, divergence, change over time). Ego-motion is always present -> use *changes* in flow.
4. **30 frames -> 1 prob**: mean/max/std pooling; consecutive-embedding differences; small 1D-conv/transformer; MIL (clip positive if ANY part is).
5. **Heads/loss**: logistic regression, LightGBM, BCE, **pairwise ranking loss** (crash clip must outscore its twin) = optimises AUC and cancels the scene.
6. **ROC-AUC**: P(random positive outranks random negative). Rank-only -> rank-average ensembles.
7. **Validation under shift**: GroupKFold (leakage), scene-cluster hold-out, adversarial validation (train-vs-test classifier reveals shortcuts).
8. **Shortcut learning + augmentation**: mask overlays, colour jitter, blur, random crop, random JPEG, h-flip, random temporal crop. No time reversal.

## Self-check results (Sep 29)
- "Danger detector" fails because train negatives are already tense; the model must detect the event, not the mood. (needs reinforcing)
- Twin structure => scene has zero label signal in train. (half right)
- Rank vs prob averaging: need to explain in own words. (not yet)
- 0.95 CV vs 0.62 LB: leakage (got it) + domain shift (missed).
