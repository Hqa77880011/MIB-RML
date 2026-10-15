import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, f1_score

from .data import corrupt, read_split
from .model import MIBRML, probabilities, representation_loss, symmetric_kl


CONDITIONS = {
    "clean": (0, 0, None),
    "missing_0.1": (0.1, 0, None),
    "missing_0.3": (0.3, 0, None),
    "missing_0.5": (0.5, 0, None),
    "missing_0.7": (0.7, 0, None),
    "weak": (0, 0.1, 20),
    "moderate": (0, 0.3, 10),
    "severe": (0, 0.5, 5),
    "joint": (0.3, 0.3, 10),
}


def batch(data, indices, device):
    streams = [torch.from_numpy(x[indices]).to(device) for x in data["x"]]
    lengths = [torch.from_numpy(length[indices]).to(device) for length in data["lengths"]]
    target = torch.from_numpy(data["y"][indices]).to(device)
    return streams, lengths, target


def calibration_error(probability, labels, multilabel=False):
    if multilabel:
        confidence = np.maximum(probability, 1 - probability).reshape(-1)
        correct = ((probability >= 0.5) == labels).reshape(-1)
    else:
        confidence = probability.max(1)
        correct = probability.argmax(1) == labels
    bins = np.minimum((confidence * 15).astype(int), 14)
    return float(sum(np.mean(bins == b) * abs(correct[bins == b].mean() - confidence[bins == b].mean())
                     for b in range(15) if np.any(bins == b)))


def score(output, target, task, classes):
    prediction, probability = output
    if task == "sentiment":
        binary = (target >= 0).astype(int)
        predicted_binary = probability.argmax(1)
        correlation = pearsonr(target, prediction[:, 0]).statistic if len(target) > 1 and np.std(target) and np.std(prediction[:, 0]) else None
        return dict(acc2=float(accuracy_score(binary, predicted_binary)),
                    f1=float(f1_score(binary, predicted_binary, average="weighted", zero_division=0)),
                    mae=float(np.mean(np.abs(target - prediction[:, 0]))),
                    pearson=float(correlation) if correlation is not None else None,
                    ece=calibration_error(probability, binary))
    if task == "multilabel":
        predicted = probability >= 0.5
        return dict(macro_f1=float(f1_score(target, predicted, average="macro", zero_division=0)),
                    micro_f1=float(f1_score(target, predicted, average="micro", zero_division=0)),
                    ece=calibration_error(probability, target, True))
    return dict(accuracy=float(accuracy_score(target, probability.argmax(1))),
                macro_f1=float(f1_score(target, probability.argmax(1), labels=range(classes), average="macro", zero_division=0)),
                ece=calibration_error(probability, target))


@torch.no_grad()
def evaluate(model, data, task, classes, args, condition, seed):
    model.eval()
    missing, text_drop, snr = CONDITIONS[condition]
    collected, probabilities_out, weights_out, availability_out = [], [], [], []
    uncertainty_out, rate_out = [], []
    with torch.random.fork_rng(devices=[torch.device(args.device).index or 0] if str(args.device).startswith("cuda") else []):
        torch.manual_seed(seed)
        for start in range(0, len(data["y"]), args.batch_size):
            indices = np.arange(start, min(start + args.batch_size, len(data["y"])))
            streams, lengths, _ = batch(data, indices, args.device)
            degraded, availability = corrupt(streams, lengths, missing, text_drop, snr, args.text_modality)
            output = model(degraded, lengths, availability)
            collected.append(output["prediction"].cpu().numpy())
            probabilities_out.append(probabilities(output["prediction"], task).cpu().numpy())
            weights_out.append(output["weights"].cpu().numpy())
            availability_out.append(availability.cpu().numpy())
            uncertainty_out.append(output["log_variance"].mean(-1).cpu().numpy())
            rate_out.append(output["rate"].cpu().numpy())
    prediction = np.concatenate(collected)
    probability = np.concatenate(probabilities_out)
    weights = np.concatenate(weights_out)
    availability = np.concatenate(availability_out)
    uncertainty = np.concatenate(uncertainty_out)
    rate = np.concatenate(rate_out)
    metrics = score((prediction, probability), data["y"], task, classes)
    metrics["mean_gate"] = weights.mean(0).tolist()
    metrics["mean_rate"] = rate.mean(0).tolist()
    denominator = availability.sum(0).clip(1)
    metrics["observed_mean_uncertainty"] = ((uncertainty * availability).sum(0) / denominator).tolist()
    return metrics, dict(prediction=prediction, probability=probability, gate=weights,
                         availability=availability, uncertainty=uncertainty, rate=rate,
                         y=data["y"], id=data["id"])


def train(args):
    if min(args.epochs, args.batch_size, args.patience) < 1 or args.lr <= 0:
        raise ValueError("Positive training budgets and learning rate required")
    if args.rho < 0 or min(args.temperature, args.label_temperature) <= 0:
        raise ValueError("Require nonnegative rho and positive contrastive temperatures")
    root = Path(args.data)
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    task, classes = metadata["task"], metadata["classes"]
    if task not in ("sentiment", "classification", "multilabel"):
        raise ValueError("Task must be sentiment, classification or multilabel")
    if task != "sentiment" and (not isinstance(classes, int) or classes < 2):
        raise ValueError("Classification and multi-label metadata require classes >= 2")
    splits = {name: read_split(root / f"{name}.npz", metadata["modalities"]) for name in ("train", "val", "test")}
    for data in splits.values():
        if not np.isfinite(data["y"]).all():
            raise ValueError("Targets must be finite")
        if task == "classification" and (data["y"].ndim != 1 or not np.isin(data["y"], np.arange(classes)).all()):
            raise ValueError("Classification targets must be class IDs")
        if task == "multilabel" and (data["y"].shape[1:] != (classes,) or not np.isin(data["y"], [0, 1]).all()):
            raise ValueError("Multi-label targets must be binary [N, classes]")
        if task == "sentiment" and data["y"].ndim != 1:
            raise ValueError("Sentiment targets must be scalar [N]")
        data["y"] = data["y"].astype(np.int64 if task == "classification" else np.float32)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    outputs = 3 if task == "sentiment" else classes
    model = MIBRML(metadata["dimensions"], outputs, args.variant, args.rho).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best, stale = -np.inf, 0
    start_time = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(len(splits["train"]["y"]))
        losses = []
        for start in range(0, len(order), args.batch_size):
            streams, lengths, target = batch(splits["train"], order[start:start + args.batch_size], args.device)
            clean, observed = corrupt(streams, lengths)
            if args.variant == "early_fusion":
                degraded, availability = clean, observed
            elif args.variant == "moddrop":
                degraded, availability = corrupt(streams, lengths, missing=0.3)
            else:
                missing, text_drop, snr = CONDITIONS[("clean", "missing_0.3", "moderate", "joint")[int(rng.integers(4))]]
                degraded, availability = corrupt(streams, lengths, missing, text_drop, snr, args.text_modality)
            optimizer.zero_grad()
            first = model(clean, lengths, observed)
            second = model(degraded, lengths, availability)
            loss = 0.5 * (representation_loss(first, target, observed, task, epoch, args)
                          + representation_loss(second, target, availability, task, epoch, args))
            if args.variant not in ("no_consistency", "early_fusion", "moddrop"):
                loss = loss + args.consistency_weight * symmetric_kl(
                    probabilities(first["prediction"], task), probabilities(second["prediction"], task), task)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        validation = {condition: evaluate(model, splits["val"], task, classes, args, condition, args.seed + 10000)[0]
                      for condition in ("clean", "missing_0.3", "moderate", "joint")}
        metric = "f1" if task == "sentiment" else "macro_f1"
        validation_score = sum(weight * validation[condition][metric] for weight, condition in
                               zip((0.4, 0.2, 0.2, 0.2), ("clean", "missing_0.3", "moderate", "joint")))
        row = dict(epoch=epoch, loss=float(np.mean(losses)), validation=validation, validation_score=validation_score)
        history.append(row)
        print(json.dumps(row), flush=True)
        if validation_score > best:
            best, stale = validation_score, 0
            torch.save(dict(state=copy.deepcopy(model.state_dict()), epoch=epoch,
                            validation_score=best, metadata=metadata, args=vars(args)), output_dir / "best.pt")
        else:
            stale += 1
        (output_dir / "history.json").write_text(json.dumps(history, indent=2, allow_nan=False), encoding="utf-8")
        if stale >= args.patience:
            break
    checkpoint = torch.load(output_dir / "best.pt", map_location=args.device, weights_only=True)
    model.load_state_dict(checkpoint["state"])
    metrics = {}
    for condition in CONDITIONS:
        metrics[condition], arrays = evaluate(model, splits["test"], task, classes, args, condition, args.seed + 20000)
        np.savez_compressed(output_dir / f"predictions_{condition}.npz", **arrays)
    result = dict(config=vars(args), metadata=metadata, best_epoch=checkpoint["epoch"],
                  validation_score=checkpoint["validation_score"], conditions=metrics,
                  elapsed_seconds=time.perf_counter() - start_time,
                  parameters=sum(p.numel() for p in model.parameters()))
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")


def evaluate_checkpoint(checkpoint_path, data_path, output, device):
    import argparse

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    args = argparse.Namespace(**checkpoint["args"])
    args.device = device
    metadata = checkpoint["metadata"]
    prepared = json.loads((Path(data_path) / "metadata.json").read_text(encoding="utf-8"))
    for key in ("dimensions", "modalities", "task", "classes"):
        if metadata[key] != prepared[key]:
            raise ValueError(f"Checkpoint/data mismatch: {key}")
    data = read_split(Path(data_path) / "test.npz", metadata["modalities"])
    size = 3 if metadata["task"] == "sentiment" else metadata["classes"]
    model = MIBRML(metadata["dimensions"], size, args.variant, args.rho).to(device)
    model.load_state_dict(checkpoint["state"])
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    metrics = {}
    for condition in CONDITIONS:
        metrics[condition], arrays = evaluate(model, data, metadata["task"], metadata["classes"],
                                              args, condition, args.seed + 20000)
        np.savez_compressed(destination / f"predictions_{condition}.npz", **arrays)
    result = dict(config=vars(args), metadata=metadata, conditions=metrics,
                  checkpoint=str(checkpoint_path), evaluation_data=str(data_path))
    (destination / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
