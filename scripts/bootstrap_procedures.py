from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


SELECTORS = ("uniform", "codec", "clinical")
BUDGETS = (1568, 3136, 6272)
SEEDS = (17, 29, 43)
COMPARISONS = (("codec", "uniform"), ("clinical", "uniform"), ("clinical", "codec"))


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"Missing prediction export: {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def metric(rows: list[dict], name: str) -> float:
    target = [int(row["target"]) for row in rows]
    prediction = [int(row["prediction"]) for row in rows]
    if name == "accuracy":
        return float(accuracy_score(target, prediction))
    return float(f1_score(target, prediction, labels=list(range(10)),
                          average="macro", zero_division=0))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paired bootstrap over held-out procedures, conditional on trained seeds")
    parser.add_argument("--runs", type=Path, default=Path("cataract101_runs"))
    parser.add_argument("--output", type=Path, default=Path("reports/procedure_bootstrap.csv"))
    parser.add_argument("--replicates", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    args = parser.parse_args()

    runs: dict[tuple[str, int, int], dict[str, list[dict]]] = {}
    procedure_sets = []
    for selector in SELECTORS:
        for budget in BUDGETS:
            for seed in SEEDS:
                path = args.runs / f"{selector}_budget{budget}_seed{seed}" / "predictions.jsonl"
                rows = load_jsonl(path)
                grouped: dict[str, list[dict]] = defaultdict(list)
                for row in rows:
                    grouped[row["video_id"]].append(row)
                runs[(selector, budget, seed)] = dict(grouped)
                procedure_sets.append(set(grouped))
    procedures = sorted(set.intersection(*procedure_sets))
    if any(set(grouped) != set(procedures) for grouped in runs.values()):
        raise SystemExit("Prediction exports do not contain identical held-out procedures")

    rng = np.random.default_rng(args.bootstrap_seed)
    output_rows = []
    for budget in BUDGETS:
        for metric_name in ("accuracy", "macro_f1"):
            samples: dict[str, list[float]] = {selector: [] for selector in SELECTORS}
            for _ in range(args.replicates):
                sampled = rng.choice(procedures, size=len(procedures), replace=True)
                for selector in SELECTORS:
                    seed_values = []
                    for seed in SEEDS:
                        grouped = runs[(selector, budget, seed)]
                        rows = [row for video_id in sampled for row in grouped[str(video_id)]]
                        seed_values.append(metric(rows, metric_name))
                    samples[selector].append(float(np.mean(seed_values)))
            for selector in SELECTORS:
                values = np.asarray(samples[selector])
                output_rows.append({
                    "visual_tokens": budget,
                    "metric": metric_name,
                    "comparison": selector,
                    "estimate": float(values.mean()),
                    "bootstrap_ci95_low": float(np.quantile(values, 0.025)),
                    "bootstrap_ci95_high": float(np.quantile(values, 0.975)),
                    "procedures": len(procedures),
                    "replicates": args.replicates,
                })
            for first, second in COMPARISONS:
                differences = np.asarray(samples[first]) - np.asarray(samples[second])
                output_rows.append({
                    "visual_tokens": budget,
                    "metric": metric_name,
                    "comparison": f"{first}_minus_{second}",
                    "estimate": float(differences.mean()),
                    "bootstrap_ci95_low": float(np.quantile(differences, 0.025)),
                    "bootstrap_ci95_high": float(np.quantile(differences, 0.975)),
                    "procedures": len(procedures),
                    "replicates": args.replicates,
                })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    metadata = {
        "interpretation": "procedure bootstrap conditional on the three trained models",
        "procedures": procedures,
        "replicates": args.replicates,
        "bootstrap_seed": args.bootstrap_seed,
        "output": str(args.output),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
