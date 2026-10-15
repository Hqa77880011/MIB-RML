import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from .data import prepare
from .model import VARIANTS
from .run import CONDITIONS, evaluate_checkpoint, train


def summarize(paths, output, resamples):
    experiments = [json.loads(Path(path).read_text(encoding="utf-8")) for path in paths]
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    grouped = {}
    for run in experiments:
        method = run["config"]["variant"]
        seed = run["config"]["seed"]
        if seed in grouped.setdefault(method, {}):
            raise ValueError("Supply only one selected configuration per method and seed")
        grouped[method][seed] = run
    statistics = []
    paired = []
    rng = np.random.default_rng(0)
    for method, runs in grouped.items():
        for condition in CONDITIONS:
            metrics = list(next(iter(runs.values()))["conditions"][condition])
            for metric in metrics:
                values = [r["conditions"][condition][metric] for r in runs.values()]
                values = [value for value in values if value is not None and np.isscalar(value)]
                if values:
                    statistics.append(dict(method=method, condition=condition, metric=metric,
                                           mean=float(np.mean(values)), sd=float(np.std(values, ddof=1)) if len(values) > 1 else None,
                                           seeds=len(values)))
        if method == "mibrml" or "mibrml" not in grouped:
            continue
        common = sorted(set(runs) & set(grouped["mibrml"]))
        if len(common) < 2:
            continue
        for condition in CONDITIONS:
            for metric in ("acc2", "f1", "mae", "pearson", "ece", "accuracy", "macro_f1", "micro_f1"):
                pairs = [(grouped["mibrml"][seed]["conditions"][condition].get(metric),
                          runs[seed]["conditions"][condition].get(metric)) for seed in common]
                if any(a is None or b is None for a, b in pairs):
                    continue
                difference = np.array([a - b for a, b in pairs])
                boot = rng.choice(difference, (resamples, len(difference)), replace=True).mean(1)
                pvalue = float(wilcoxon(difference).pvalue) if np.any(difference) else 1.0
                paired.append(dict(baseline=method, condition=condition, metric=metric,
                                   mean_difference=float(difference.mean()),
                                   ci95=np.quantile(boot, [0.025, 0.975]).tolist(), pvalue=pvalue, seeds=common))
    previous = 0
    for index, row in enumerate(sorted(paired, key=lambda r: r["pvalue"])):
        previous = max(previous, min(1, (len(paired) - index) * row["pvalue"]))
        row["holm_pvalue"] = previous
    (destination / "summary.json").write_text(json.dumps(statistics, indent=2, allow_nan=False), encoding="utf-8")
    (destination / "paired.json").write_text(json.dumps(paired, indent=2, allow_nan=False), encoding="utf-8")


def plot(paths, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    runs = [json.loads(Path(path).read_text(encoding="utf-8")) for path in paths]
    methods = sorted({r["config"]["variant"] for r in runs})
    metric = "f1" if runs[0]["metadata"]["task"] == "sentiment" else "macro_f1"
    fig, ax = plt.subplots(figsize=(7, 4))
    conditions = ["clean", "missing_0.1", "missing_0.3", "missing_0.5", "missing_0.7"]
    for method in methods:
        values = [[r["conditions"][condition][metric] for r in runs if r["config"]["variant"] == method]
                  for condition in conditions]
        ax.errorbar([0, 0.1, 0.3, 0.5, 0.7], [np.mean(v) for v in values],
                    yerr=[np.std(v, ddof=1) if len(v) > 1 else 0 for v in values], marker="o", label=method)
    ax.set(xlabel="Nominal modality-drop probability", ylabel=metric)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(destination / "missingness.png", dpi=180)
    plt.close(fig)
    for condition in ("clean", "moderate", "joint"):
        run = next((r for r in runs if r["config"]["variant"] == "mibrml"), runs[0])
        fig, axes = plt.subplots(1, 2, figsize=(8, 3))
        axes[0].bar(range(len(run["conditions"][condition]["mean_gate"])), run["conditions"][condition]["mean_gate"], color="#518393")
        axes[1].bar(range(len(run["conditions"][condition]["mean_rate"])), run["conditions"][condition]["mean_rate"], color="#6b9a80")
        axes[0].set(xlabel="Modality", ylabel="Mean gate")
        axes[1].set(xlabel="Modality", ylabel="Mean rate multiplier")
        fig.tight_layout()
        fig.savefig(destination / f"reliability_{condition}.png", dpi=180)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(prog="python -m mibrml")
    commands = parser.add_subparsers(dest="command", required=True)
    data = commands.add_parser("prepare")
    data.add_argument("--manifest", required=True)
    data.add_argument("--output", required=True)
    run = commands.add_parser("train")
    run.add_argument("--data", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--variant", choices=VARIANTS, default="mibrml")
    run.add_argument("--device", default="cpu")
    run.add_argument("--seed", type=int, default=13)
    run.add_argument("--epochs", type=int, default=60)
    run.add_argument("--batch-size", type=int, default=32)
    run.add_argument("--patience", type=int, default=8)
    run.add_argument("--lr", type=float, default=0.0001)
    run.add_argument("--weight-decay", type=float, default=0.0001)
    run.add_argument("--rho", type=float, default=0.5)
    run.add_argument("--beta", type=float, default=0.01)
    run.add_argument("--auxiliary-weight", type=float, default=0.2)
    run.add_argument("--alignment-weight", type=float, default=0.1)
    run.add_argument("--covariance-weight", type=float, default=0.01)
    run.add_argument("--consistency-weight", type=float, default=0.5)
    run.add_argument("--temperature", type=float, default=0.1)
    run.add_argument("--label-temperature", type=float, default=1)
    run.add_argument("--text-modality", type=int, default=0)
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--checkpoint", required=True)
    evaluation.add_argument("--data", required=True)
    evaluation.add_argument("--output", required=True)
    evaluation.add_argument("--device", default="cpu")
    summary = commands.add_parser("summarize")
    summary.add_argument("--metrics", nargs="+", required=True)
    summary.add_argument("--output", default="outputs/statistics")
    summary.add_argument("--resamples", type=int, default=10000)
    visual = commands.add_parser("plot")
    visual.add_argument("--metrics", nargs="+", required=True)
    visual.add_argument("--output", default="outputs/figures")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.manifest, args.output)
    elif args.command == "train":
        train(args)
    elif args.command == "evaluate":
        evaluate_checkpoint(args.checkpoint, args.data, args.output, args.device)
    elif args.command == "summarize":
        summarize(args.metrics, args.output, args.resamples)
    else:
        plot(args.metrics, args.output)


if __name__ == "__main__":
    main()
