# MIB-RML

Implementation of **Reliability-Conditioned Information Bottleneck for Robust Multimodal Learning under Missing and Noisy Modalities**.

MIB-RML uses a stochastic posterior for each modality. Posterior uncertainty allocates bottleneck pressure and reduces fusion trust, while shared/private representations and paired observation views provide complementary training signals. The repository supports scalar sentiment, multiclass classification and multilabel classification on supplied sequence features.

## Install

Use Python 3.11 or 3.12 and install from the repository root:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

Activate the environment before installation (`source .venv/bin/activate`, or `.venv\Scripts\Activate.ps1` in PowerShell). Install a CUDA-compatible PyTorch build first when training on GPU. Every training command accepts `--device cpu` or `--device cuda`.

## 1. Obtain feature datasets

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

## 2. Normalize and prepare

```bash
python -m mibrml prepare --manifest data/manifest.json --output data/prepared
```

Preparation checks example IDs across splits, excludes padding and missing streams from normalization, and uses only training-split feature means/standard deviations. It writes normalized `train.npz`, `val.npz`, `test.npz` and `metadata.json`. Keep official speaker, subject and recording partitions when exporting source features; IDs detect duplicate examples but cannot infer missing grouping metadata.

## 3. Train

```bash
python -m mibrml train --data data/prepared --output outputs/mibrml_13 --variant mibrml --seed 13 --device cuda
```

Each modality uses a 128-dimensional projection, sinusoidal positions, a two-layer Transformer with four heads and a 256-unit feed-forward block. Pooled observed tokens parameterize a diagonal Gaussian with a 64-dimensional latent. Shared/private projections have 32 dimensions each. Training samples latent noise; evaluation uses posterior means.

RCRA normalizes `exp(rho * mean_log_variance)` over observed modalities and scales it by their count. The weighted KL therefore redistributes rate pressure while preserving its mean scale. The fusion score subtracts a nonnegative uncertainty coefficient from a learned latent-mean score. Absent modalities receive zero rate multiplier and zero gate weight. The objective combines task/auxiliary supervision, bottleneck KL, label-aware cross-modal alignment, shared/private covariance and symmetric paired-view KL.

Defaults follow the paper's stated training budget: AdamW, learning rate 0.0001, weight decay 0.0001, batch size 32, gradient clipping at 1, at most 60 epochs and patience 8. Bottleneck pressure ramps linearly over ten epochs. One clean and one sampled degraded view share all parameters. Checkpoint selection uses `0.4*clean_F1 + 0.2*missing_0.3_F1 + 0.2*moderate_F1 + 0.2*joint_F1`; test conditions use that single frozen checkpoint.

| Argument | Default | Meaning |
|---|---:|---|
| `--epochs`, `--patience`, `--batch-size` | 60, 8, 32 | Optimization and early-stopping budget |
| `--lr`, `--weight-decay` | 0.0001, 0.0001 | AdamW settings |
| `--rho` | 0.5 | Reliability-conditioned rate allocation strength |
| `--beta` | 0.01 | Bottleneck KL coefficient |
| `--auxiliary-weight` | 0.2 | Per-observed-modality supervision |
| `--alignment-weight` | 0.1 | Label-aware cross-modal agreement |
| `--covariance-weight` | 0.01 | Shared/private covariance penalty |
| `--consistency-weight` | 0.5 | Clean/degraded predictive KL |
| `--temperature` | 0.1 | Alignment similarity temperature |
| `--label-temperature` | 1 | Label-distance denominator weighting |
| `--text-modality` | 0 | Text stream index; use −1 for sensor/audio/video-only features |

Sentiment supervision uses an absolute-error regression head and a two-class head with negative versus nonnegative labels. The latter supplies predictive distributions for consistency and calibration. Classification uses cross-entropy; multilabel prediction uses binary cross-entropy. These task-head choices make the predictive-distribution interface explicit where the paper leaves it unspecified.

## 4. Compare components and feature-level baselines

```bash
python -m mibrml train --data data/prepared --output outputs/no_rcra_13 --variant no_rcra --seed 13 --device cuda
python -m mibrml train --data data/prepared --output outputs/moddrop_13 --variant moddrop --seed 13 --device cuda
```

| Variant | Change |
|---|---|
| `mibrml` | Complete method |
| `early_fusion` | Concatenates deterministic modality codes, trained on available clean inputs |
| `moddrop` | Same deterministic concatenation with modality-drop training |
| `uniform_ib` | Uniform rate allocation and uniform observed-modality fusion |
| `no_rcra` | Uniform rate allocation, learned reliability fusion |
| `no_gate` | RCRA with uniform observed-modality fusion |
| `no_shared_private` | Direct weighted latent fusion, without alignment/covariance decomposition losses |
| `no_alignment` | Removes cross-modal alignment |
| `no_consistency` | Removes paired predictive consistency |
| `no_auxiliary` | Removes unimodal auxiliary supervision |

The comparison baselines are defined by their implemented mechanisms above. They do not relabel these compact feature baselines as MulT, MISA, TATE or other separately published architectures. Use the same feature exports, splits, validation rule and seeds across comparisons. The paper's five seeds are 13/29/47/71/97. Hyperparameter trials are controlled by the listed CLI arguments; preserve the same 20-trial budget when performing the paper's search protocol.

## 5. Missing and noisy inputs

Final evaluation runs the following fixed conditions:

| Condition | Modality-drop probability | Text vector drop | Other-stream SNR |
|---|---:|---:|---:|
| `clean` | 0 | 0 | unchanged |
| `missing_0.1/0.3/0.5/0.7` | 0.1/0.3/0.5/0.7 | 0 | unchanged |
| `weak` | 0 | 0.1 | 20 dB |
| `moderate` | 0 | 0.3 | 10 dB |
| `severe` | 0 | 0.5 | 5 dB |
| `joint` | 0.3 | 0.3 | 10 dB |

Independent modality masks are resampled when every available stream would be removed. Text vector masking preserves the first and final valid tokens as special-token positions. Gaussian noise is scaled by each sample's valid feature power. Noise and drop operations affect observed streams only. Masks and noise use a fixed condition-evaluation seed shared across variants. Keep evaluation batch size fixed across comparisons, since random draws are generated batch by batch.

These commands operate on features. Pixel-level face occlusion, raw-video frame deletion and raw sensor-axis transformations must be applied before feature extraction when studying the corresponding CREMA-D/UCI input-space protocols. Store their extracted feature states in separate prepared dataset directories, preserving the original training split for identical normalization, and evaluate the frozen checkpoint:

```bash
python -m mibrml evaluate --checkpoint outputs/mibrml_13/best.pt --data data/occluded_prepared --output outputs/occluded_13 --device cuda
```

The evaluator checks task/modality/feature dimensions and applies the same condition table to the supplied test features. Its `clean` row measures the supplied externally transformed state without additional feature corruption. The built-in noise table provides the feature-level controlled study; it does not label feature noise as a pixel-level intervention.

## 6. Read and summarize outputs

`best.pt` stores the selected model, metadata, epoch and configuration. `history.json` records training loss and all four validation conditions. `metrics.json` contains each final test condition, mean gate/rate diagnostics, selected epoch, parameter count and elapsed time. Each `predictions_<condition>.npz` stores predictions, probabilities, targets, IDs, availability, gate, uncertainty and rate allocation for inspectable per-example analysis.

Sentiment outputs are Acc-2, weighted binary F1, MAE, Pearson correlation and 15-bin ECE. Acc-2/F1/ECE use the binary head; MAE/correlation use the scalar regression head. Classification outputs accuracy, Macro-F1 and ECE. Multilabel outputs macro/micro-F1 and elementwise ECE with threshold 0.5. Undefined correlation is `null`; results are fractions rather than percentages.

```bash
python -m mibrml summarize --metrics outputs/mibrml_13/metrics.json outputs/mibrml_29/metrics.json outputs/no_rcra_13/metrics.json outputs/no_rcra_29/metrics.json --output outputs/statistics
python -m mibrml plot --metrics outputs/mibrml_13/metrics.json outputs/no_rcra_13/metrics.json --output outputs/figures
```

Summaries report seed means and sample standard deviations. When at least two seeds match, paired statistics report MIB-RML minus baseline, 10,000 paired bootstrap intervals, Wilcoxon p-values and Holm correction across the supplied endpoint family. MAE/ECE favor negative differences; accuracy/F1/correlation favor positive differences. Supply one selected configuration per method/seed from the same dataset and feature protocol.

Plots show missingness curves with seed error bars and modality-level reliability/rate diagnostics. Predictions and diagnostics are generated from actual execution outputs.
