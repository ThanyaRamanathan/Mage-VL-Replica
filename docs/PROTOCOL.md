# Cataract-101 experimental protocol

## Registered question

At a fixed visual-token budget, do native-codec or phase-supervised sparse patch
selectors outperform uniform temporal sampling for cataract phase recognition,
and does the supervised selector outperform the codec selector?

## Data and leakage control

- Use the 101 original Cataract-101 H.264/AVC files and official transitions.
- Freeze the seeded 71/15/15 procedure split emitted by the converter.
- Keep all clips from a procedure in exactly one partition.
- Fit clinical importance only on training procedures.
- Select checkpoints on validation and evaluate the test set once.
- Publish the split JSON, source hashes, package versions, and seeds.

## Matched budgets

A 224×224 frame contains a 14×14 grid of 16×16 patches (196 tokens). Primary
budgets are 1,568 / 3,136 / 6,272, equal to 8 / 16 / 32 dense frames. Every
method emits the requested count. Padding and selector operations are excluded
from visual tokens and reported separately.

Codec and clinical arms use identical native keyframe anchors. Dense anchors are
mandatory; a run fails if anchors exceed the budget. Only predicted-frame scores
differ between these arms.

## Models and supervision

Use a patch-16 pretrained ViT with native selected-patch projection and a common
temporal classifier. Begin with frozen spatial blocks. Fine-tuning is a separate
ablation.

The clinical scorer receives phase and ±5-second transition supervision, plus
spatial total-variation and selection-rate penalties. This is weak domain
supervision, not causal clinical localization.

## Metrics

- Accuracy, macro F1, and per-phase precision/recall/F1
- Accuracy within ±5 seconds of a transition
- Synchronized model-forward milliseconds per sample from held-out evaluation
- Exact visual tokens for every method and budget
- Mean and standard deviation across seeds 17, 29, and 43

The completed experiment records mean model-forward time only. It does not
record warmup-controlled median/p95, decode/selector/packing breakdowns, or peak
CUDA memory. Those remain explicitly out of scope for the reported latency
claim and must not be inferred from `latency_ms`.

## Analysis

Plot accuracy and macro F1 against tokens and milliseconds. Report per-phase and
transition-distance strata. Pre-register qualitative cases using the largest
codec-minus-clinical and clinical-minus-codec loss differences plus random ties.

## Claim boundary

Cataract-101 provides original H.264/AVC High-profile video. Motion vectors are
codec-native. Residual importance is decoded motion-compensated L1 error, not
bit-exact transform-coefficient energy, because stable PyAV does not expose the
coefficients. The report must use this wording.

## Completion gate

- Unit tests and a native H.264 smoke test pass.
- Every clip satisfies exact-budget and anchor invariants.
- Three selectors × three budgets × three seeds complete.
- Test is evaluated only from validation-selected checkpoints.
- Results include split hashes, metrics, forward timings, per-phase analysis,
  and selector-disagreement cases.
- Measured results are clearly separated from plans and estimates.
