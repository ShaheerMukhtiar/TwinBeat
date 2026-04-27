# WiFi CSI-Based Contactless Heart Rate Estimation Pipeline

A Python pipeline that estimates human heart rate (BPM) from **WiFi Channel State Information (CSI)** data — without any wearable sensor. The system reads CSI recordings stored as CSV files, applies signal processing and spectral analysis, and outputs a predicted heart rate along with a confidence score.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [How It Works — End-to-End](#how-it-works--end-to-end)
  - [1. Data Loading & Validation](#1-data-loading--validation)
  - [2. Ground-Truth Extraction](#2-ground-truth-extraction)
  - [3. CSI Matrix Extraction](#3-csi-matrix-extraction)
  - [4. Chunking](#4-chunking)
  - [5. Signal Processing Pipeline (per chunk)](#5-signal-processing-pipeline-per-chunk)
  - [6. Chunk-Level Prediction Fusion](#6-chunk-level-prediction-fusion)
  - [7. Global Summary](#7-global-summary)
- [Signal Processing Pipeline — Deep Dive](#signal-processing-pipeline--deep-dive)
  - [Preprocessing](#preprocessing)
  - [Subcarrier Selection](#subcarrier-selection)
  - [Wavelet Decomposition & Bandpass Filtering](#wavelet-decomposition--bandpass-filtering)
  - [Frequency Analysis (Welch PSD)](#frequency-analysis-welch-psd)
  - [Histogram Voting](#histogram-voting)
  - [Confidence Scoring](#confidence-scoring)
  - [Temporal Smoothing](#temporal-smoothing)
- [Data Flow Diagram](#data-flow-diagram)
- [Input Format](#input-format)
- [Output](#output)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Key Design Decisions](#key-design-decisions)

---

## Overview

When a person breathes or their heart beats, the tiny chest-wall movements modulate the WiFi signal reflected off their body. These modulations are captured in the **CSI subcarrier amplitudes** reported by commodity WiFi NICs. This pipeline extracts that periodic breathing/cardiac signal and converts it to a BPM estimate.

| Property            | Value                          |
|---------------------|--------------------------------|
| Sampling rate (fs)  | **100 Hz** (Data Collected)     |
| Frequency band      | **0.8 – 2.5 Hz** (48 – 150 BPM) |
| Subcarriers used    | Top **10** (selected per chunk)|
| Chunk size          | **2000 samples** (~100 s)      |
| BPM output range    | **50 – 130 BPM** (clamped)     |

---

## Project Structure

```
Python_pipline/
├── main_mutChunks.py   # Entry point — file loading, chunking, fusion, reporting
├── Pipeline.py         # Core signal processing — subcarrier selection → BPM estimate
├── Cal_dt/             # Input folder containing CSV files with CSI data
└── README.md           # This file
```

---

## How It Works — End-to-End

### 1. Data Loading & Validation

**File:** `main_mutChunks.py → safe_read_csv()`

The program scans the `Cal_dt/` folder for all `*.csv` files. Each file is loaded using a fault-tolerant reader that attempts multiple encodings (`utf-8`, then `latin1`) and falls back to the Python CSV engine with bad-line skipping. Files that cannot be parsed at all are skipped with a log message.

### 2. Ground-Truth Extraction

**File:** `main_mutChunks.py → extract_gt_from_filename()`

The ground-truth heart rate is embedded in each filename using the pattern `GT_<value>` (e.g., `subject1_GT_72.5_sitting.csv` → GT = 72.5 BPM). A regex extracts this value. Files without a valid GT pattern are skipped.

### 3. CSI Matrix Extraction

**File:** `main_mutChunks.py → extract_csi_matrix()`

All columns in the CSV are coerced to numeric values. Columns that are entirely non-numeric (e.g., timestamps, labels) are dropped. The result is a 2-D NumPy array of shape **(N_samples × N_subcarriers)**, with any remaining NaNs replaced by zero.

### 4. Chunking

**File:** `main_mutChunks.py → chunk_array()`

Long recordings are split into non-overlapping chunks of **2000 rows** each (≈100 seconds at 20 Hz). Chunks smaller than **200 rows** are discarded to avoid unreliable short-window estimates. Each chunk is processed independently through the signal pipeline.

### 5. Signal Processing Pipeline (per chunk)

**File:** `Pipeline.py → pipeline()`

Each chunk is passed to the core `pipeline()` function, which returns:
- **`pred`** — estimated heart rate in BPM
- **`conf`** — confidence score in [0, 1]

_(Detailed breakdown in the [Deep Dive](#signal-processing-pipeline--deep-dive) section below.)_

### 6. Chunk-Level Prediction Fusion

**File:** `main_mutChunks.py → process_file()`

After all chunks for a file are processed, the per-chunk predictions are fused into a single final prediction using **robust outlier filtering**:

1. Compute the **median** of all chunk predictions.
2. Compute the **Median Absolute Deviation (MAD)**.
3. Discard any prediction farther than **2.5 × MAD** from the median.
4. The final prediction is the **mean** of the remaining (filtered) values.

This prevents a single noisy chunk from skewing the result.

### 7. Global Summary

**File:** `main_mutChunks.py → main()`

After all files are processed, a global summary is printed:

- Number of files processed
- Mean / Std / Min / Max absolute error against ground truth

---

## Signal Processing Pipeline — Deep Dive

All steps below happen inside `Pipeline.py → pipeline()` for a single CSI chunk.

### Preprocessing

```
Raw CSI chunk (N × M)
    │
    ├─ Remove dead subcarriers (all-zero columns)
    ├─ Trim 5 seconds from each edge (remove transients)
    ├─ Mean-center each subcarrier (zero-mean)
    └─ Smooth each subcarrier with a 3-point centered moving average
```

- **Edge trimming** removes setup/teardown artifacts.
- **Mean centering** eliminates DC offset, leaving only the AC (breathing/cardiac) component.
- **Smoothing** reduces high-frequency noise while preserving the breathing/cardiac waveform.

### Subcarrier Selection

**Function:** `select_subcarriers()`

Not all WiFi subcarriers carry useful cardiac information. The pipeline selects the **top 10** subcarriers using a composite score:

```
score = 0.5 × normalized_variance + 0.5 × correlation_stability
```

| Metric                   | Purpose                                                        |
|--------------------------|----------------------------------------------------------------|
| **Normalized variance**  | Captures subcarriers with strong signal modulation             |
| **Correlation stability**| Measures how well each subcarrier correlates with the mean signal across all subcarriers |

> **Design note:** Using raw energy (sum of squares) for selection caused a bias toward ~95 BPM in earlier versions. The variance + correlation approach avoids this "attractor" effect.

### Wavelet Decomposition & Bandpass Filtering

**Function:** `wavelet_band()`

Each selected subcarrier is processed through:

1. **Discrete Wavelet Transform (DWT)** using the `db4` wavelet at up to 4 decomposition levels.
2. **Selective reconstruction** — only detail coefficients at levels 1–3 are reconstructed. This isolates the mid-to-high frequency components where cardiac signals reside, while discarding low-frequency drift (approximation coefficients) and very high-frequency noise.
3. **Butterworth bandpass filter** (3rd order, 0.8–2.5 Hz) is applied to the reconstructed signal, strictly limiting the output to the physiological heart rate range.

If wavelet decomposition fails for any reason, the function falls back to bandpass filtering alone.

### Frequency Analysis (Welch PSD)

For each of the 10 selected subcarriers:

1. Compute the **Power Spectral Density (PSD)** using Welch's method (`nperseg = min(256, signal_length)`).
2. Isolate the **0.8 – 2.5 Hz** band (heart rate range).
3. Detect **spectral peaks** (with prominence ≥ 8% of the band maximum).
4. Keep the **top 2 peaks** per subcarrier.
5. Convert peak frequency to BPM (`bpm = freq × 60`).
6. Discard peaks outside **55 – 130 BPM**.
7. Weight each candidate by its PSD power × a **harmonic penalty**:
   ```
   harmonic_penalty(bpm) = exp(-|bpm - 85| / 40)
   ```
   This gently favors estimates near a resting heart rate of ~85 BPM and penalizes physiologically unlikely extremes.

The result is a list of weighted BPM candidates (typically 10–20 values from all subcarriers).

### Histogram Voting

**Key innovation to remove the 95 BPM bias.**

Instead of taking a simple weighted average of all BPM candidates (which can be pulled toward outliers):

1. Build a **weighted histogram** with 25 bins spanning 60–120 BPM.
2. Select the **bin with the highest total weight**.
3. The final BPM is the **center of that bin**.

This acts as a robust mode estimator, naturally ignoring scattered outlier candidates.

### Confidence Scoring

```
confidence = (0.6 × count_factor + 0.4 × spread_factor) × harmonic_penalty(final_bpm)
```

| Component        | Formula                              | Meaning                                         |
|------------------|--------------------------------------|-------------------------------------------------|
| Count factor     | `min(1.0, num_estimates / 10)`       | More agreeing subcarriers → higher confidence   |
| Spread factor    | `1 / (1 + std(estimates))`           | Tighter estimate spread → higher confidence     |
| Harmonic penalty | `exp(-\|bpm - 85\| / 40)`            | Physiologically normal range → higher confidence|

Output is clamped to **[0, 1]**.

### Temporal Smoothing

If a previous chunk's BPM estimate (`LAST_BPM`) exists and the current confidence exceeds **0.4**:

```
final_bpm = 0.6 × LAST_BPM + 0.4 × current_bpm
```

This exponential moving average prevents sudden jumps between consecutive chunks and provides physiologically plausible continuity. The final BPM is clamped to **[50, 130]**.

---

## Data Flow Diagram

```
Cal_dt/*.csv
    │
    ▼
┌──────────────────────────────┐
│  safe_read_csv()             │  Fault-tolerant CSV loading
│  extract_gt_from_filename()  │  GT from filename regex
│  extract_csi_matrix()        │  Coerce to numeric matrix
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│  chunk_array()               │  Split into 2000-row chunks
│  (min 200 rows per chunk)    │
└──────────────┬───────────────┘
               │
               ▼  (per chunk)
┌──────────────────────────────────────────────────────────┐
│  Pipeline.py → pipeline(csi_chunk)                       │
│                                                          │
│  1. Remove dead subcarriers                              │
│  2. Trim edges (5s each side)                            │
│  3. Mean-center + smooth                                 │
│  4. Select top 10 subcarriers (variance + stability)     │
│  5. Wavelet decomposition (db4) + bandpass (0.8–2.5 Hz)  │
│  6. Welch PSD → peak detection → BPM candidates          │
│  7. Harmonic penalty weighting                           │
│  8. Histogram voting → final BPM                         │
│  9. Confidence scoring                                   │
│ 10. Temporal smoothing with LAST_BPM                     │
│                                                          │
│  Returns: (predicted_bpm, confidence)                    │
└──────────────────────────┬───────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────┐
│  Robust Fusion               │  Median + MAD outlier filter
│  → Final per-file BPM        │  → Mean of inliers
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│  Global Summary              │  Mean/Std/Min/Max error
│  across all files            │  across all processed files
└──────────────────────────────┘
```

---

## Input Format

### CSV Files

- Place all CSI recording files in the **`Cal_dt/`** folder.
- Each CSV should contain **numeric CSI amplitude data** (one row per time sample, one column per subcarrier).
- Non-numeric columns (timestamps, labels, etc.) are automatically dropped.
- The **filename must contain `GT_<value>`** to indicate the ground-truth heart rate (e.g., `recording_GT_75.0_rest.csv`).

### Assumptions

- **Sampling rate:** 20 Hz (hardcoded as `fs = 20.0` in `Pipeline.py`).
- **Minimum data:** At least 100 rows (5 seconds) of valid CSI data per file, and at least 200 rows per chunk.

---

## Output

### Per-Chunk Output

```
Chunk 01 | Pred:   72.5 | GT:   73.0 | Err:    0.5 | Conf: 0.78 ✓
Chunk 02 | Pred:   74.1 | GT:   73.0 | Err:    1.1 | Conf: 0.81 ✓
```

| Symbol | Meaning            |
|--------|--------------------|
| ✓      | Error < 5 BPM      |
| ○      | Error 5 – 10 BPM   |
| ✗      | Error > 10 BPM     |

### Per-File Summary

```
Final Result
Predicted HR: 73.2
GT HR:        73.0
Error:        0.2
Chunks used:  4
```

### Global Summary

```
GLOBAL SUMMARY
Files: 12
Mean Error: 3.45
Std Error:  2.18
Min Error:  0.20
Max Error:  9.80
```

---

## Dependencies

| Package       | Purpose                                    |
|---------------|--------------------------------------------|
| `numpy`       | Array operations and numerical computation |
| `pandas`      | CSV loading and data manipulation          |
| `scipy`       | Butterworth filter, Welch PSD, peak finding|
| `pywt`        | Discrete Wavelet Transform (PyWavelets)    |

### Install

```bash
pip install numpy pandas scipy PyWavelets
```

---

## Usage

1. Place your CSI CSV files in the `Cal_dt/` folder (filenames must contain `GT_<value>`).
2. Run:

```bash
python main_mutChunks.py
```

3. The pipeline will process all CSV files and print per-chunk, per-file, and global results to the console.

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Variance + correlation** for subcarrier selection (not raw energy) | Raw energy selection collapsed estimates toward ~95 BPM regardless of actual heart rate |
| **Histogram voting** instead of weighted mean | Weighted mean was vulnerable to harmonic peaks and outlier subcarriers pulling the estimate |
| **Wavelet + bandpass** dual filtering | Wavelet decomposition isolates the cardiac band more cleanly than bandpass alone; bandpass serves as a hard-limit safety net |
| **MAD-based outlier filtering** for chunk fusion | Standard deviation is sensitive to outliers; MAD is robust to up to 50% corrupted chunks |
| **Temporal smoothing** (EMA with LAST_BPM) | Heart rate is physiologically continuous; prevents noisy chunk-to-chunk jumps |
| **Harmonic penalty centered at 85 BPM** | Acts as a soft prior for resting heart rate; exponential decay still allows detection across the full 50–130 BPM range |
| **Chunk size of 2000 / min 200** | 2000 samples (100s) gives sufficient frequency resolution; chunks below 200 (10s) produce unreliable PSD estimates |
