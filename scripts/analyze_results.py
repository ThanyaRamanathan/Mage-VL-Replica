from __future__ import annotations

import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path


SELECTORS = ("uniform", "codec", "clinical")
BUDGETS = (1568, 3136, 6272)
SEEDS = (17, 29, 43)
COLORS = {"uniform": "#2563eb", "codec": "#f97316", "clinical": "#16a34a"}
RUN_PATTERN = re.compile(
    r"^(uniform|codec|clinical)_budget(1568|3136|6272)_seed(17|29|43)$"
)
T_CRIT_DF2 = 4.302652729749


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean_sd(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values)


def paired_stats(values: list[float]) -> dict[str, float]:
    mean, sd = mean_sd(values)
    se = sd / math.sqrt(len(values))
    half_width = T_CRIT_DF2 * se
    if se == 0:
        p_value = 1.0 if mean == 0 else 0.0
    else:
        t_value = abs(mean / se)
        # Closed-form two-sided survival probability for Student-t with df=2.
        p_value = 1.0 - t_value / math.sqrt(t_value * t_value + 2.0)
    return {
        "mean_difference": mean,
        "sd_difference": sd,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
        "paired_t_p_exploratory": p_value,
    }


def load_results(root: Path) -> dict[tuple[str, int, int], dict]:
    results: dict[tuple[str, int, int], dict] = {}
    for directory in root.iterdir():
        if not directory.is_dir() or not RUN_PATTERN.fullmatch(directory.name):
            continue
        path = directory / "results.json"
        if not path.exists():
            raise SystemExit(f"Canonical run is missing results.json: {directory}")
        item = json.loads(path.read_text(encoding="utf-8"))
        key = (str(item["selector"]), int(item["visual_tokens"]), int(item["seed"]))
        name_key = RUN_PATTERN.fullmatch(directory.name)
        assert name_key is not None
        expected = (name_key.group(1), int(name_key.group(2)), int(name_key.group(3)))
        if key != expected:
            raise SystemExit(f"Metadata mismatch in {path}: {key} != {expected}")
        if key in results:
            raise SystemExit(f"Duplicate canonical result: {key}")
        results[key] = item

    expected_keys = {(s, b, seed) for s in SELECTORS for b in BUDGETS for seed in SEEDS}
    missing = sorted(expected_keys - set(results))
    extra = sorted(set(results) - expected_keys)
    if missing or extra:
        raise SystemExit(f"Result grid mismatch. Missing={missing}; extra={extra}")
    return results


def svg_line_chart(
    path: Path,
    title: str,
    y_label: str,
    series: dict[str, list[tuple[int, float, float]]],
    y_min: float,
    y_max: float,
) -> None:
    width, height = 900, 520
    left, right, top, bottom = 90, 30, 65, 75
    plot_w, plot_h = width - left - right, height - top - bottom

    def x_pos(value: int) -> float:
        return left + (value - BUDGETS[0]) / (BUDGETS[-1] - BUDGETS[0]) * plot_w

    def y_pos(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Inter,Segoe UI,Arial,sans-serif;fill:#172033}'
        '.axis{stroke:#64748b;stroke-width:1}.grid{stroke:#e2e8f0;stroke-width:1}'
        '.tick{font-size:13px}.title{font-size:22px;font-weight:700}'
        '.label{font-size:15px;font-weight:600}.legend{font-size:14px}</style>',
        f'<text x="{width / 2}" y="32" text-anchor="middle" class="title">{title}</text>',
    ]
    for index in range(6):
        value = y_min + index * (y_max - y_min) / 5
        y = y_pos(value)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" class="grid"/>')
        parts.append(f'<text x="{left-12}" y="{y+5:.1f}" text-anchor="end" class="tick">{value:.2f}</text>')
    parts.extend(
        [
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" class="axis"/>',
            f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" class="axis"/>',
        ]
    )
    for budget in BUDGETS:
        x = x_pos(budget)
        parts.append(f'<text x="{x:.1f}" y="{height-bottom+28}" text-anchor="middle" class="tick">{budget:,}</text>')
    parts.append(f'<text x="{left+plot_w/2}" y="{height-18}" text-anchor="middle" class="label">Visual patch tokens</text>')
    parts.append(
        f'<text transform="translate(24 {top+plot_h/2}) rotate(-90)" text-anchor="middle" class="label">{y_label}</text>'
    )

    for selector in SELECTORS:
        points = series[selector]
        color = COLORS[selector]
        coordinates = " ".join(f"{x_pos(x):.1f},{y_pos(y):.1f}" for x, y, _ in points)
        parts.append(f'<polyline points="{coordinates}" fill="none" stroke="{color}" stroke-width="3"/>')
        for budget, mean, sd in points:
            x, y = x_pos(budget), y_pos(mean)
            upper, lower = y_pos(min(y_max, mean + sd)), y_pos(max(y_min, mean - sd))
            parts.append(f'<line x1="{x:.1f}" y1="{upper:.1f}" x2="{x:.1f}" y2="{lower:.1f}" stroke="{color}" stroke-width="2"/>')
            parts.append(f'<line x1="{x-6:.1f}" y1="{upper:.1f}" x2="{x+6:.1f}" y2="{upper:.1f}" stroke="{color}" stroke-width="2"/>')
            parts.append(f'<line x1="{x-6:.1f}" y1="{lower:.1f}" x2="{x+6:.1f}" y2="{lower:.1f}" stroke="{color}" stroke-width="2"/>')
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{color}"/>')

    legend_x = left + 15
    for index, selector in enumerate(SELECTORS):
        x = legend_x + index * 150
        parts.append(f'<line x1="{x}" y1="{top+14}" x2="{x+25}" y2="{top+14}" stroke="{COLORS[selector]}" stroke-width="3"/>')
        parts.append(f'<text x="{x+32}" y="{top+19}" class="legend">{selector.title()}</text>')
    parts.append('</svg>')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")


def svg_scatter(path: Path, summary: list[dict]) -> None:
    width, height = 900, 520
    left, right, top, bottom = 90, 35, 65, 75
    plot_w, plot_h = width - left - right, height - top - bottom
    x_min, x_max = 25.0, 145.0
    y_min, y_max = 0.69, 0.775

    def x_pos(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * plot_w

    def y_pos(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Inter,Segoe UI,Arial,sans-serif;fill:#172033}'
        '.axis{stroke:#64748b}.grid{stroke:#e2e8f0}.tick{font-size:13px}'
        '.title{font-size:22px;font-weight:700}.label{font-size:15px;font-weight:600}'
        '.pointlabel{font-size:12px}</style>',
        f'<text x="{width/2}" y="32" text-anchor="middle" class="title">Accuracy–latency trade-off</text>',
    ]
    for value in range(40, 141, 20):
        x = x_pos(value)
        parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{height-bottom}" class="grid"/>')
        parts.append(f'<text x="{x:.1f}" y="{height-bottom+27}" text-anchor="middle" class="tick">{value}</text>')
    for value in (0.70, 0.72, 0.74, 0.76):
        y = y_pos(value)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" class="grid"/>')
        parts.append(f'<text x="{left-12}" y="{y+5:.1f}" text-anchor="end" class="tick">{value:.2f}</text>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" class="axis"/>')
    parts.append(f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" class="axis"/>')
    parts.append(f'<text x="{left+plot_w/2}" y="{height-18}" text-anchor="middle" class="label">Mean latency (ms/sample)</text>')
    parts.append(f'<text transform="translate(24 {top+plot_h/2}) rotate(-90)" text-anchor="middle" class="label">Mean accuracy</text>')
    for row in summary:
        selector = str(row["selector"])
        x, y = x_pos(float(row["latency_ms_mean"])), y_pos(float(row["accuracy_mean"]))
        radius = {1568: 6, 3136: 8, 6272: 10}[int(row["visual_tokens"])]
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius}" fill="{COLORS[selector]}" fill-opacity="0.82"/>')
        prefix = {"uniform": "U", "codec": "C", "clinical": "L"}[selector]
        parts.append(f'<text x="{x+10:.1f}" y="{y-8:.1f}" class="pointlabel">{prefix}{int(row["visual_tokens"])//196}f</text>')
    parts.append('</svg>')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    results = load_results(project / "cataract101_runs")
    reports = project / "reports"
    figures = reports / "figures"

    long_rows = []
    for key in sorted(results, key=lambda x: (x[1], x[0], x[2])):
        item = results[key]
        long_rows.append(
            {
                "selector": key[0],
                "visual_tokens": key[1],
                "seed": key[2],
                "accuracy": item["accuracy"],
                "macro_f1": item["macro_f1"],
                "transition_window_accuracy": item["transition_window_accuracy"],
                "latency_ms": item["latency_ms"],
                "validation_best_macro_f1": item["validation_best_macro_f1"],
            }
        )
    write_csv(reports / "cataract101_results_long.csv", long_rows)

    summary = []
    for budget in BUDGETS:
        for selector in SELECTORS:
            values = [results[(selector, budget, seed)] for seed in SEEDS]
            row: dict[str, str | int | float] = {
                "selector": selector,
                "visual_tokens": budget,
                "seeds": len(values),
            }
            for metric in ("accuracy", "macro_f1", "transition_window_accuracy", "latency_ms"):
                mean, sd = mean_sd([float(value[metric]) for value in values])
                row[f"{metric}_mean"] = mean
                row[f"{metric}_std"] = sd
            summary.append(row)
    write_csv(reports / "cataract101_summary.csv", summary)

    comparisons = (("codec", "uniform"), ("clinical", "uniform"), ("clinical", "codec"))
    paired_rows = []
    for budget in BUDGETS:
        for first, second in comparisons:
            for metric in ("accuracy", "macro_f1", "transition_window_accuracy", "latency_ms"):
                differences = [
                    float(results[(first, budget, seed)][metric])
                    - float(results[(second, budget, seed)][metric])
                    for seed in SEEDS
                ]
                stats = paired_stats(differences)
                paired_rows.append(
                    {
                        "visual_tokens": budget,
                        "comparison": f"{first}_minus_{second}",
                        "metric": metric,
                        **stats,
                        "seed17_difference": differences[0],
                        "seed29_difference": differences[1],
                        "seed43_difference": differences[2],
                    }
                )
    write_csv(reports / "paired_differences.csv", paired_rows)

    per_class_rows = []
    for budget in BUDGETS:
        for selector in SELECTORS:
            class_values: dict[int, list[float]] = defaultdict(list)
            supports: dict[int, set[int]] = defaultdict(set)
            for seed in SEEDS:
                for item in results[(selector, budget, seed)]["per_class"]:
                    class_id = int(item["class_id"])
                    class_values[class_id].append(float(item["f1"]))
                    supports[class_id].add(int(item["support"]))
            for class_id in sorted(class_values):
                mean, sd = mean_sd(class_values[class_id])
                per_class_rows.append(
                    {
                        "selector": selector,
                        "visual_tokens": budget,
                        "class_id": class_id,
                        "f1_mean": mean,
                        "f1_std": sd,
                        "test_support": ";".join(map(str, sorted(supports[class_id]))),
                    }
                )
    write_csv(reports / "per_class_f1.csv", per_class_rows)

    for metric, title, label, y_min, y_max, filename in (
        ("accuracy", "Accuracy versus visual-token budget", "Test accuracy", 0.68, 0.78, "accuracy_vs_tokens.svg"),
        ("macro_f1", "Macro-F1 versus visual-token budget", "Macro-F1", 0.62, 0.73, "macro_f1_vs_tokens.svg"),
        ("transition_window_accuracy", "Transition-window accuracy versus budget", "Accuracy within ±5 s", 0.52, 0.67, "transition_vs_tokens.svg"),
    ):
        chart_series: dict[str, list[tuple[int, float, float]]] = defaultdict(list)
        for row in summary:
            chart_series[str(row["selector"])].append(
                (int(row["visual_tokens"]), float(row[f"{metric}_mean"]), float(row[f"{metric}_std"]))
            )
        svg_line_chart(figures / filename, title, label, chart_series, y_min, y_max)
    svg_scatter(figures / "accuracy_latency.svg", summary)

    manifest = {
        "canonical_results": len(results),
        "selectors": list(SELECTORS),
        "budgets": list(BUDGETS),
        "seeds": list(SEEDS),
        "outputs": [
            "cataract101_results_long.csv",
            "cataract101_summary.csv",
            "paired_differences.csv",
            "per_class_f1.csv",
            "figures/accuracy_vs_tokens.svg",
            "figures/macro_f1_vs_tokens.svg",
            "figures/transition_vs_tokens.svg",
            "figures/accuracy_latency.svg",
        ],
    }
    (reports / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
