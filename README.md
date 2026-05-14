# neon-gaze-angle

A reproducible pipeline for processing and analyzing **gaze angle** from a [Pupil Neon](https://pupil-labs.com/products/neon) eye tracker. Designed for open-science workflows: every processing step runs in a Jupyter notebook so you can inspect intermediate results in real time.

## What it does

The pipeline combines IMU head-pitch data with gaze-elevation data to compute a **gaze angle** (where a person is looking relative to the ground plane). It then segments continuous recordings into individual walking trials using Pupil Neon annotations, and runs **Recurrence Quantification Analysis (RQA)** on each trial to characterize gaze dynamics.

## Repository structure

```
neon-gaze-angle/
├── neon_gaze/              # Shared Python package (importable utilities)
│   ├── io.py               # Load/save Pupil Neon CSV exports
│   ├── processing.py       # Synchronize IMU + gaze, compute gaze angle, mask blinks
│   ├── gait.py             # Detect heel-strike / toe-off from IMU accelerometry
│   ├── segmentation.py     # Parse trial annotations, segment into per-trial CSVs
│   ├── rqa.py              # Recurrence quantification analysis
│   ├── plotting.py         # Plotly and Matplotlib visualization helpers
│   └── visualization.py    # Interactive Bokeh recurrence widget
│
├── notebooks/              # Analysis notebooks (run these in order)
│   ├── 01_compute_gaze_angle.ipynb
│   ├── 02_segment_trials.ipynb
│   ├── 03_analyze_rqa.ipynb
│   ├── 04_recurrence_visualization.ipynb
│   └── explore_fixations.ipynb   # Standalone exploratory notebook
│
├── demo/                   # Small sample data for the walkthrough
│   ├── input/              # Trimmed imu.csv and gaze_positions.csv
│   └── walk_segmented_csvs/  # Two demo trial segments
│
├── requirements.txt
├── LICENSE                 # MIT
└── README.md
```

Directories not tracked in git (created by the pipeline at runtime):

- `data/` — your full experimental data (Pupil Neon exports, segmented trials)
- `output/` — pipeline outputs (RQA summaries, generated HTML widgets)

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

The pipeline requires Python 3.10+ and the following packages: `numpy`, `pandas`, `scipy`, `plotly`, `matplotlib`, `ipywidgets`, `bokeh`.

### 2. Run the demo

Open **`notebooks/01_compute_gaze_angle.ipynb`** in Jupyter. The default configuration points at the included demo data (`demo/input/`), so you can **Run All** immediately to see the pipeline produce a gaze-angle time series, a histogram, and gait-event overlays.

### 3. Use your own data

Each notebook has a **Configuration** cell at the top. Change `DATA_DIR` to point at your own Pupil Neon export folder, set `SAVE_OUTPUT = True`, and run the notebook.

## Pipeline walkthrough

### Notebook 01 — Compute Gaze Angle

**Input:** `imu.csv` + `gaze_positions.csv` from a Pupil Neon export.

Synchronizes the two data streams (gaze is downsampled to IMU frame rate via nearest-timestamp matching) and computes `gaze angle = pitch + elevation`. Produces interactive plots of the time series and its distribution.

**Output:** `gaze_angle.csv` with columns `timestamp [ns]`, `gaze angle [deg]`, `pitch [deg]`, `elevation [deg]`, `time_sec`.

### Notebook 02 — Segment Trials

**Input:** A Pupil Neon export that includes `annotations.csv` (with "Trial Begin"/"Trial End" labels) and `blinks.csv`.

Computes gaze angle, masks blink intervals with NaN, parses trial annotations, and splits the recording into one CSV per trial. Odd-numbered trials are labelled Walk-Dir-Up, even are Walk-Dir-Down.

**Output:** Per-trial CSVs in `data/walk_segmented_csvs/`, named like `Subject-03_Hill-Condition_Trial-01_Walk-Dir-Up.csv`.

### Notebook 03 — RQA Analysis

**Input:** Folder of segmented trial CSVs.

For each trial, the gaze-angle series is z-scored, then analyzed with Recurrence Quantification Analysis. Two modes are available: fixed-epsilon (same radius for all trials) or RR-locked (radius chosen per trial to match a target recurrence rate). Also computes autocorrelation statistics (mean ACF, AUC).

**Output:** A summary CSV with one row per trial, containing descriptive stats and all RQA measures (RR, DET, MaxL, ENT, L, LAM, TT).

### Notebook 04 — Recurrence Visualization

**Input:** A single trial CSV.

Generates a standalone Bokeh HTML widget with linked-brushing between the gaze time series and its recurrence plot. Click any point to highlight all of its recurrent partners in both panels.

**Output:** `output/recurrence_widget.html`.

### Explore Fixations (standalone)

Loads two `fixations.csv` files and compares fixation-duration distributions between conditions. Not part of the main pipeline.

## Data formats

### Input (Pupil Neon exports)

| File | Key columns |
|------|-------------|
| `imu.csv` | `timestamp [ns]`, `pitch [deg]`, `yaw [deg]`, `acceleration z [G]` |
| `gaze_positions.csv` | `timestamp [ns]`, `elevation [deg]`, `azimuth [deg]` |
| `blinks.csv` | `start timestamp [ns]`, `end timestamp [ns]` |
| `annotations.csv` | `timestamp [ns]`, `label` (values: `"Trial Begin"`, `"Trial End"`) |
| `fixations.csv` | `start timestamp [ns]`, `end timestamp [ns]`, `duration [ms]` |

### Output (segmented trial CSVs)

| Column | Description |
|--------|-------------|
| `timestamp [ns]` | Original Pupil Neon nanosecond timestamp |
| `gaze angle [deg]` | Head pitch + gaze elevation |
| `pitch [deg]` | IMU head pitch |
| `elevation [deg]` | Gaze elevation relative to head |
| `time_sec` | Seconds since start of recording |
| `trial event` | `"Trial N Start"` or `"Trial N End"` (sparse) |
| `trial` | Trial number |

## Design principles

The codebase follows SOLID programming principles adapted for a research context:

- **Single Responsibility:** Each module in `neon_gaze/` handles one concern (I/O, processing, gait detection, segmentation, RQA, plotting). Notebooks orchestrate these modules without reimplementing logic.
- **Open/Closed:** Processing functions accept parameters (epsilon, target RR, filter cutoffs) without requiring internal changes. New analysis modes can be added by writing new notebooks that import the same modules.
- **Dependency Inversion:** Notebooks depend on the `neon_gaze` package interface, not on implementation details. You can swap out the RQA engine or plotting library without rewriting notebooks.
- **DRY:** Functions that were previously copy-pasted across multiple notebooks (e.g. `compute_rqa`, `parse_filename`, `build_recurrence_plot_widget`) now live in one place.

## License

MIT
