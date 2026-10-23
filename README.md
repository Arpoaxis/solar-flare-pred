# Solar Flare Prediction Using Deep Learning

**Active-region forecasting of ≥M1.0 solar flares within 24 hours**

A senior capstone project developed at **California State University, Dominguez Hills (CSUDH)**. This project investigates whether convolutional neural networks (CNNs) and gated recurrent units (GRUs) can use multiwavelength solar observations to forecast strong solar flares. It emphasizes reproducible data preparation, leakage-resistant evaluation, rare-event detection, and probabilistic forecast quality.

**Author:** Gerard F. McCallion  
**Project:** CSC 492 Senior Design  
**Primary tools:** Python, PyTorch, torchvision, NumPy, pandas, and CUDA

## Project Overview

Solar flares are sudden releases of magnetic energy in the Sun's atmosphere. Strong events can disrupt radio communications, satellite operations, and other technologies affected by space weather. Forecasting them is challenging because significant flares are rare, active regions evolve over time, and evaluation can be misleading if related observations are divided between training and testing.

This project frames forecasting as **binary classification at the solar active-region (AR) level**:

> Given images of a solar active region, estimate the probability that it will produce at least one **M1.0-class or stronger flare during the next 24 hours**.

An M1.0 flare corresponds to a peak GOES 1–8 Å soft X-ray flux of **1.0 × 10⁻⁵ W/m²**. The project compares models that use a single observation with models that learn from a sequence of four observations.

### Project Highlights

- Built an image-based pipeline using the **SDOBenchmark** dataset and observations from NASA's **Solar Dynamics Observatory (SDO)**.
- Retained **8,310 complete samples**, each containing **40 images** (4 timesteps × 10 observation channels).
- Created fixed **active-region-disjoint** train, validation, and test sets, verifying **zero active-region overlap**.
- Evaluated models under both **AR-disjoint** and **chronological holdout** protocols.
- Compared five CNN-based spatial and temporal model configurations.
- Evaluated both flare discrimination and forecast probability quality using **PR-AUC, True Skill Statistic (TSS), and Brier score**.
- Trained models using GPU-accelerated PyTorch on an **NVIDIA RTX 3080**.

## Dataset

**Source:** [SDOBenchmark](https://i4ds.github.io/SDOBenchmark/)

The data comprise active-region cutouts from two SDO instruments:

| Instrument | Observation channels |
|---|---|
| Atmospheric Imaging Assembly (AIA) | 94, 131, 171, 193, 211, 304, 335, and 1700 Å |
| Helioseismic and Magnetic Imager (HMI) | Continuum and magnetogram |

Each complete sample has:

- **4 timesteps** preceding the forecast period.
- **10 image channels** at each timestep.
- **40 grayscale JPG images** in total, each **256 × 256 pixels**.
- A model input tensor with shape **`[T=4, C=10, H=256, W=256]`**.
- A binary label: **`1`** for at least one ≥M1.0 flare in the next 24 hours; **`0`** otherwise.

Images are loaded as floating-point tensors and rescaled to **[0, 1]**. An integrity filter excludes incomplete samples so that every retained input has the same dimensions and channel coverage.

### Filtered Dataset and AR-Disjoint Splits

The complete-sample index covers observations from **January 2012 through December 2017**.

| Partition | Samples | Positive labels | Distinct active regions |
|---|---:|---:|---:|
| Training | 6,704 | 429 | 959 |
| Validation | 807 | 44 | 106 |
| Test | 799 | 152 | 89 |
| **Total** | **8,310** | **625** | **1,154** |

Each active region is assigned to only one partition. The intersections between training, validation, and test active-region IDs were checked and found to be **zero**. The differing positive-label frequencies across partitions are an important consideration when interpreting results.

A separate **chronological holdout** evaluation was also used to test performance in a more future-like setting, rather than relying on a single split design.

## Models

The final study compares five configurations, progressing from spatial-only image classification to models that incorporate temporal information and class weighting.

| Model | Purpose |
|---|---|
| **CNN baseline** | Uses the last observed timestep to learn spatial features from 10 channels. |
| **CNN baseline + positive class weighting** | Tests how weighting the rare flare class affects the spatial baseline. |
| **CNN + temporal mean pooling** | Encodes all four timesteps and averages their extracted features without recurrence. |
| **CNN + GRU** | Extracts CNN features from four timesteps and uses a GRU to model temporal evolution. |
| **CNN + GRU + positive class weighting** | Combines temporal modeling with imbalance-aware training; early stopping is used in the chronological comparison. |

For temporal models, the conceptual flow is:

```text
4 observation times × 10 image channels
                 |
        CNN feature extraction
                 |
     Sequence of feature vectors
                 |
       Mean pooling or GRU
                 |
        Binary classification
                 |
  P(≥M1.0 flare in the next 24 h)
```

## Training and Evaluation

Model training uses **binary cross-entropy with logits**. Selected configurations use positive-class weighting to address the rarity of significant flares. Early stopping is used in the designated experiments.

To avoid tuning on the test set:

1. Models are trained on the training partition.
2. Checkpoints are selected using validation performance.
3. For threshold-dependent evaluation, each model's decision threshold is selected to maximize **validation TSS**.
4. That fixed validation-selected threshold is applied to the corresponding test partition.
5. The same evaluation procedure is repeated for the chronological holdout protocol.

### Metrics

- **PR-AUC (Precision–Recall Area Under the Curve):** Assesses how effectively the model distinguishes the rare positive flare class across decision thresholds; higher is better.
- **TSS (True Skill Statistic):** Measures true positive rate minus false positive rate; higher is better.
- **Brier score:** Measures the squared error of forecast probabilities; **lower is better**.

Accuracy alone is not emphasized because predicting the majority "no-flare" class can produce superficially high accuracy while missing significant solar events.

## Results

The results below are from the project's final paper. **All values are test-set metrics**. TSS uses each model's operating threshold selected on its corresponding validation set.

### Active-Region-Disjoint Evaluation

| Model | Test PR-AUC ↑ | Test TSS ↑ | Test Brier ↓ |
|---|---:|---:|---:|
| CNN (last timestep) | 0.5623 | 0.3570 | 0.1263 |
| CNN (last timestep) + positive weighting | 0.6037 | 0.5446 | 0.2373 |
| CNN + temporal mean pooling (all 4 timesteps) | 0.6149 | 0.4607 | **0.1167** |
| CNN + GRU (all 4 timesteps) | 0.5715 | 0.5113 | 0.1228 |
| **CNN + GRU + positive weighting** | **0.6379** | **0.5678** | 0.1849 |

The **weighted CNN + GRU** achieved the highest test PR-AUC and TSS in this setting. **Temporal mean pooling** achieved the lowest test Brier score.

### Chronological Holdout Evaluation

| Model | Test PR-AUC ↑ | Test TSS ↑ | Test Brier ↓ |
|---|---:|---:|---:|
| CNN (last timestep) | 0.3657 | 0.5856 | 0.0509 |
| CNN (last timestep) + positive weighting | 0.3961 | 0.6753 | 0.1117 |
| CNN + temporal mean pooling + early stopping | 0.4041 | 0.5911 | 0.0487 |
| CNN + GRU | 0.4176 | 0.6760 | **0.0467** |
| **CNN + GRU + positive weighting + early stopping** | **0.4377** | **0.6777** | 0.1211 |

The **weighted CNN + GRU with early stopping** again achieved the highest test PR-AUC and TSS, while the **unweighted CNN + GRU** produced the lowest Brier score. Results changed substantially between evaluation protocols, reinforcing the importance of how train/test partitions are constructed.

### Key Finding

**Temporal models generally improved flare discrimination**, especially when combined with positive-class weighting. However, stronger detection performance did **not** always produce better-calibrated or more accurate probabilities. Evaluating PR-AUC, TSS, and Brier score together provides a more complete picture than selecting a model using a single metric.

## Repository Organization

The project is organized around reusable model and data code, experiment scripts, intermediate metadata, and locally stored experiment outputs.

```text
solar-flare-pred/
├── README.md
├── src/                           # Dataset, models, and reusable utilities
├── scripts/                       # Data, training, testing, and evaluation scripts
├── data/
│   ├── external/
│   │   └── sdobenchmark/
│   │       └── raw/               # Raw dataset (not committed)
│   └── interim/
│       └── sdobenchmark/
│           ├── index.parquet      # Generated sample metadata
│           └── splits/
│               ├── train.csv
│               ├── val.csv
│               └── test.csv
└── runs/                          # Local checkpoints and experiment outputs
```

The layout above illustrates the principal project directories; additional scripts and generated files may be present. Large source images, model checkpoints, and other experiment artifacts are intentionally kept outside the GitHub repository where appropriate.

## Local Setup and Reproduction

### Requirements

- **Python 3.12** (development environment: Python 3.12.10)
- **PyTorch** and **torchvision** (CUDA-enabled builds recommended for GPU training)
- **NumPy, pandas, Pillow, and PyArrow**
- Additional Python dependencies required by the repository's training and evaluation scripts
- The **SDOBenchmark** raw data, downloaded separately

The project was developed on **Windows 11** using an **NVIDIA GeForce RTX 3080 (10 GB)**. Other supported environments may work, but were not the primary test environment.

### Prepare the Environment (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install a version of PyTorch appropriate for your CPU/GPU platform using the [official PyTorch installation instructions](https://pytorch.org/get-started/locally/), then install the other packages required by the project scripts. Refer to a dependency file in the repository if one is provided.

### Reproduce the Workflow

1. Download the SDOBenchmark dataset from the source linked above and place it under the locally configured raw-data directory.
2. Configure the dataset paths in the project scripts to match your machine; do not assume the original developer's Windows drive paths are available.
3. Build or load the filtered sample index and verify the expected **4 × 10 = 40 images per sample**.
4. Use the provided frozen **train/validation/test split definitions** where available. Confirm no active-region ID appears in more than one split.
5. Run the model training scripts under `scripts/` and select checkpoints based on validation performance.
6. Run evaluation for both AR-disjoint and chronological protocols, selecting thresholds on validation data only.
7. Compare PR-AUC, TSS, and Brier score with the tables above.

**Reproduction note:** Exact training entrypoints and local paths depend on the scripts present in the checked-out repository. The raw images and large saved experiment outputs are not supplied by this README; downloading the dataset and running training may require substantial disk space and GPU time.

## Limitations and Future Work

This is a **research capstone**, not a deployed real-time space-weather warning service. Results are specific to the SDOBenchmark data, the selected binary target, the training configurations, and the two evaluation protocols. In particular, class imbalance, changes in the distribution of solar activity over time, and potential data/label limitations can affect generalization.

Possible extensions include:

- Post-hoc probability calibration and more detailed reliability analysis.
- Additional feature-based baselines using SHARP magnetic parameters.
- Alternative sequence models and longer observation histories.
- Additional forecasting horizons and C-, M-, or X-class prediction targets.
- Near-real-time data ingestion and monitoring for operational deployment.

## Data and Research Acknowledgment

This work uses the **SDOBenchmark** dataset and observations collected by NASA's **Solar Dynamics Observatory**, including its **AIA** and **HMI** instruments. See the [SDOBenchmark project](https://i4ds.github.io/SDOBenchmark/) for dataset information, usage guidance, and relevant citations.

This repository documents an academic research project. It is **not an official NASA or NOAA forecasting product**.
