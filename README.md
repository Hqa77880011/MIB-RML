# MIB-RML

Implementation of **Reliability-Conditioned Information Bottleneck for Robust Multimodal Learning under Missing and Noisy Modalities**.

## Installation

Use Python 3.11 or 3.12 for installation. Run from the repository root in an activated environment.

```bash
python -m pip install -r requirements.txt
```

Inspection environment: Python 3.13.5; numpy 2.1.3, scipy 1.15.3, torch 2.7.1+cu118, scikit-learn 1.6.1, matplotlib 3.10.0. Checks: Python syntax, configuration parsing and CLI/source argument matching. Training and full evaluation were not run.

## Data and preprocessing

Use the official train/validation/test partitions. CMU-MOSI and CMU-MOSEI features are available through the [CMU Multimodal SDK](https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK); CH-SIMS and aligned sentiment feature resources are documented by [MMSA](https://github.com/thuiar/MMSA). Cross-domain inputs can come from [CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D), [MM-IMDb](https://github.com/johnarevalo/gmu-mmimdb) and [UCI HAR](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones).

Export each official split as an NPZ:

| Key | Shape | Meaning |
|---|---|---|
| `x0`, `x1`, ... | `[N,T_m,D_m]` | Co-recorded modality features, padded in time |
| `length0`, `length1`, ... | `[N]` | Valid sequence lengths; zero means absent |
| `y` | `[N]` or `[N,K]` | Scalar sentiment/class ID or multilabel target |
| `id` | `[N]` strings | Globally unique example IDs across all splits |

Examples with every modality absent are rejected. Padding values must be finite and are zeroed during preparation; zero-length streams are excluded from the encoder and all observation-weighted losses. Static features use a sequence length of one. Use normalized sentiment scores from the chosen release, class IDs 0 through K−1 for classification, and binary vectors for multilabel targets. MM-IMDb targets are genre vectors; UCI HAR can use separate accelerometer and gyroscope streams; CREMA-D uses audio/video features.

The paper uses sentiment dimensions T/A/V = 768/5/20 for MOSI, 768/74/35 for MOSEI, and 768/33/709 for CH-SIMS. The code infers dimensions and supports other feature releases when the selection is kept identical across compared methods. It consumes features, so tokenization, acoustic extraction and video-backbone extraction belong to the chosen dataset release.

Create `data/manifest.json`:

```json
{
  "modalities": 3,
  "task": "sentiment",
  "train": "raw/train.npz",
  "val": "raw/val.npz",
  "test": "raw/test.npz"
}
```

For a classification task, set `"task": "classification"` and add `"classes": 6`, for example. For multilabel prediction, set `"task": "multilabel"` with the number of target labels in `classes`. Paths are relative to the manifest.

```bash
python -m mibrml prepare --manifest data/manifest.json --output data/prepared
```

Preparation checks example IDs across splits, excludes padding and missing streams from normalization, and uses only training-split feature means/standard deviations. It writes normalized `train.npz`, `val.npz`, `test.npz` and `metadata.json`. Keep official speaker, subject and recording partitions when exporting source features; IDs detect duplicate examples but cannot infer missing grouping metadata.

## Training and inference

```bash
python -m mibrml train --data data/prepared --output outputs/mibrml_13 --variant mibrml --seed 13
```

## Evaluation

Training evaluates clean, modality-missing and feature-noise conditions. Re-evaluate a frozen model with:

```bash
python -m mibrml evaluate --checkpoint outputs/mibrml_13/best.pt --data data/prepared --output outputs/evaluation_13
```

## Comparisons and ablations

```bash
python -m mibrml train --data data/prepared --output outputs/no_rcra_13 --variant no_rcra --seed 13
python -m mibrml train --data data/prepared --output outputs/moddrop_13 --variant moddrop --seed 13
```

## Outputs

`best.pt` stores the selected model; `history.json` stores losses; `metrics.json` contains task and corruption-condition metrics.

```bash
python -m mibrml plot --metrics outputs/mibrml_13/metrics.json outputs/no_rcra_13/metrics.json --output outputs/figures
```
