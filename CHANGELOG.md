# Changelog

## 0.3.0 — Methodology correction infrastructure

- Batch equal-size frame fragments to remove the per-frame encoder-call artifact.
- Add an experimental joint sparse spatio-temporal attention path.
- Align codec saliency with model sampling intervals through max/mean aggregation.
- Add clip-level clinical-importance training and optional score calibration.
- Audit realized dense-anchor counts at every token budget.
- Export per-clip predictions and bootstrap uncertainty over test procedures.
- Add validation-only selection ablations for anchor fraction, motion weight,
  temporal aggregation, and score normalization.

The published 0.2 result table remains a historical result. It was not silently
relabelled as a 0.3 result; corrected conclusions require the new experiments.

## 0.2.0 — Initial Cataract-101 study

- Three selectors at 1,568, 3,136, and 6,272 visual tokens across three training
  seeds.
- Accuracy, macro-F1, transition-window accuracy, and wall-clock measurements.
- Reproducible PBS workflows, aggregate reports, plots, and qualitative tools.
