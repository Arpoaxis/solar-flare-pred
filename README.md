# Solar Flare Prediction

Senior capstone project focused on forecasting whether a solar active region will produce a **≥ M1.0 solar flare within the next 24 hours** using deep learning and NASA Solar Dynamics Observatory data through the **SDOBenchmark** dataset.

This repository contains the project code, configuration, and small reproducibility metadata used to support model development and evaluation. Raw dataset files and large experiment outputs are intentionally excluded.

## Project Overview

Solar flares are sudden releases of magnetic energy from the Sun that can affect satellite operations, radio communications, navigation systems, and power infrastructure. This project investigates machine learning and deep learning methods for **binary solar flare forecasting**, with emphasis on:

- reproducible data handling
- leakage-safe train/validation/test design
- rare-event evaluation
- calibrated probabilistic forecasting

The primary prediction task is:

> Given observations of a solar active region, predict whether it will produce a flare of class **M1.0 or greater within 24 hours**.

## Objectives

The project is designed to compare multiple modeling approaches for the same forecasting task, including:

- classical baselines using SHARP-style magnetic features
- a CNN-only spatial baseline
- a CNN + GRU spatiotemporal model

Evaluation focuses on metrics appropriate for imbalanced event forecasting, including:

- **PR-AUC** (primary metric)
- **TSS** (True Skill Statistic)
- **Brier score**
- calibration analysis and reliability behavior


## Repository Contents

```text
.
├── configs/
├── scripts/
├── src/
├── data/
│   └── interim/
│       ├── channel_timepoint_report_training.csv
│       ├── missing_report_training.csv
│       └── sdobenchmark/
│           ├── index.parquet
│           └── splits/
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt

```


## Setup

Create and activate a virtual environment, then install dependencies:


```bash
pip install -r requirements.txt
