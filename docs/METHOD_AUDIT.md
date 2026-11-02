# Method audit and v0.3 correction plan

The completed 27-run experiment is retained as the **v0.2 isolated-selector
baseline**. It answers what happened under that implementation, but it is not a
fair test of Mage-ViT's jointly trained sparse spatio-temporal encoder. This
audit records the confounds and the changes required for v0.3.

| Concern | v0.2 status | Resolution | Requires rerun? |
|---|---|---|---:|
| Per-frame Python encoder loop | Confirmed; one ViT call per represented frame | Batch/group frame encodes for timing controls; report encoder-call count | Yes for latency |
| No spatial attention across frames | Confirmed; each frame is encoded and mean-pooled independently | Add a joint sparse spatio-temporal encoder with token-level time/space position before attention | Yes, full grid |
| Codec/model time-scale mismatch | Confirmed; source-frame saliency is based on ~40 ms change while model samples at 2 fps | Aggregate codec evidence over each interval between sampled frames, and ablate instantaneous vs interval aggregation | Yes, codec arms |
| Dense-anchor count unknown | Resolved by post-run audit | All 12,377 windows used exactly 4/8/16 anchors at 1,568/3,136/6,272 tokens; no fallback | No |
| Clinical cross-frame calibration | Confirmed; phase pooling is offset-invariant while selection is global across time | Normalize both saliency arms symmetrically and train/calibrate clinical scores on complete clips | Yes, clinical arms |
| Intervals cover training noise only | Confirmed; split is fixed | Export clip predictions with procedure IDs and perform paired procedure bootstrap | Evaluation-only pass |
| Unablated selector hyperparameters | Confirmed | Ablate `anchor_fraction`, `motion_weight`, and score normalization on validation before one held-out test | Yes, validation runs |

## Interpretation boundary for v0.2

The v0.2 result supports this narrow statement:

> With a frozen densely pretrained ViT that encodes each selected frame
> independently, uniform dense-frame sampling outperformed the tested codec and
> clinical sparse selectors on Cataract-101.

It does **not** establish that codec-native tokenization is intrinsically worse
than uniform sampling. Sparse arms were passed to an encoder whose attention
and positional structure were not trained for fragmented spatio-temporal token
sets.

## Minimum v0.3 experiment

1. Implement and smoke-test the joint sparse encoder and interval-aligned codec
   cache.
2. Use a fixed, reported anchor policy and symmetric score normalization.
3. Run seed 17 at 1,568 and 3,136 tokens for all three selectors as a gate.
4. Continue to seeds 29 and 43 only if the gate is stable.
5. Export per-clip predictions for all final checkpoints and bootstrap the 15
   held-out procedures.
6. Keep the 6,272 dense endpoint as a determinism check, not evidence about
   selector quality.

The legacy Cholec80 and HEVC helpers are not part of the Cataract-101 result and
should be moved to a clearly named `legacy/` area or removed in a later cleanup
commit.
