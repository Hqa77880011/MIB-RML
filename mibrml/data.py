import json
from pathlib import Path

import numpy as np
import torch


def read_split(path, modalities):
    with np.load(path, allow_pickle=False) as archive:
        streams = [archive[f"x{m}"].astype(np.float32) for m in range(modalities)]
        lengths = [archive[f"length{m}"].astype(np.int64) for m in range(modalities)]
        y = archive["y"]
        ids = archive["id"].astype(str)
    if len(np.unique(ids)) != len(ids) or len(y) != len(ids):
        raise ValueError("Each example needs a unique ID and target")
    for values, length in zip(streams, lengths):
        if values.ndim != 3 or len(values) != len(y) or length.shape != (len(y),):
            raise ValueError("Streams must be [N,time,features] with length vectors [N]")
        if not np.isfinite(values).all() or (length < 0).any() or (length > values.shape[1]).any():
            raise ValueError("Non-finite features or invalid sequence lengths")
        padding = np.arange(values.shape[1])[None, :] >= length[:, None]
        values[padding] = 0
    if (np.stack(lengths, 1).sum(1) == 0).any():
        raise ValueError("Every example must have at least one observed modality")
    return dict(x=streams, lengths=lengths, y=y, id=ids)


def prepare(manifest, output):
    manifest = Path(manifest).resolve()
    spec = json.loads(manifest.read_text(encoding="utf-8"))
    modalities = int(spec["modalities"])
    if modalities < 2:
        raise ValueError("At least two modalities required")
    splits = {name: read_split(manifest.parent / spec[name], modalities) for name in ("train", "val", "test")}
    all_ids = np.concatenate([split["id"] for split in splits.values()])
    if len(np.unique(all_ids)) != len(all_ids):
        raise ValueError("Sample IDs cross the official split boundaries")
    for m in range(modalities):
        train = splits["train"]
        values, lengths = train["x"][m], train["lengths"][m]
        valid = np.arange(values.shape[1])[None, :] < lengths[:, None]
        if not valid.any():
            raise ValueError(f"No observed training features for modality {m}")
        mean = values[valid].mean(0)
        standard = values[valid].std(0).clip(1e-6)
        for split in splits.values():
            current = split["x"][m]
            if current.shape[2] != values.shape[2]:
                raise ValueError("Feature dimensions differ across splits")
            current = (current - mean) / standard
            padding = np.arange(current.shape[1])[None, :] >= split["lengths"][m][:, None]
            current[padding] = 0
            split["x"][m] = current
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    for name, split in splits.items():
        payload = dict(y=split["y"], id=split["id"])
        for m in range(modalities):
            payload[f"x{m}"] = split["x"][m]
            payload[f"length{m}"] = split["lengths"][m]
        np.savez_compressed(destination / f"{name}.npz", **payload)
    (destination / "metadata.json").write_text(json.dumps(dict(modalities=modalities,
        dimensions=[x.shape[2] for x in splits["train"]["x"]], task=spec["task"],
        classes=spec.get("classes")), indent=2), encoding="utf-8")


def corrupt(streams, lengths, missing=0, text_drop=0, snr=None, text_modality=0):
    availability = torch.stack([length > 0 for length in lengths], 1).to(torch.float32)
    observed = availability.clone()
    if not 0 <= missing < 1:
        raise ValueError("Missing probability must be in [0,1)")
    pending = torch.ones(len(availability), dtype=torch.bool, device=availability.device)
    while pending.any():
        count = int(pending.sum())
        draws = streams[0].new_empty((count, len(streams))).uniform_() >= missing
        observed[pending] = draws * availability[pending]
        pending = observed.sum(1) == 0
    result = []
    for m, values in enumerate(streams):
        current = values.clone()
        valid = streams[0].new_tensor(range(values.shape[1]))[None, :] < lengths[m][:, None]
        if m == text_modality and text_drop:
            keep = streams[0].new_empty(valid.shape).uniform_() >= text_drop
            keep[:, 0] = True
            keep[torch.arange(len(keep), device=keep.device), (lengths[m] - 1).clamp_min(0)] = True
            current *= keep[:, :, None]
        elif snr is not None:
            power = (current.square() * valid[:, :, None]).sum((1, 2))
            denominator = (lengths[m] * current.shape[2]).clamp_min(1)
            scale = (power / denominator / (10 ** (snr / 10))).sqrt()
            current = current + torch.randn_like(current) * scale[:, None, None]
        current *= (valid * observed[:, m:m + 1])[:, :, None]
        result.append(current)
    return result, observed
