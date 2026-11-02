# Reporting artifacts

- [`FINAL_REPORT.md`](FINAL_REPORT.md): portfolio-ready experimental report.
- `cataract101_results_long.csv`: all 27 canonical results.
- `cataract101_summary.csv`: mean and sample SD across three seeds.
- `paired_differences.csv`: paired seed-wise selector differences with
  exploratory small-sample intervals.
- `per_class_f1.csv`: phase-level F1 summaries.
- `analysis_manifest.json`: validated factorial-design manifest.
- `figures/`: dependency-free SVG result figures.
- `qualitative/selection_cases.csv`: reproducible selection metadata.

Qualitative JPEGs are generated locally by
`python scripts/qualitative_selectors.py` and ignored by Git because they are
derived from medical video data. Confirm the dataset's redistribution terms
before publishing any frame imagery.
