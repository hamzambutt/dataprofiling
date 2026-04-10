# Data Profiling & Drift Detection Toolkit 🚀

A professional-grade Python utility designed for **Per-Column Dynamic Profiling**. This toolkit automatically adapts its statistical analysis based on data types, providing deep insights into dataset health and preparing features for drift detection. 

Developed as part of the Backend Development Internship at **Biome**.

---

## 🌟 Key Features

* **Type-Aware Branching Logic:** Automatically detects and applies specific profiling metrics for:
    * **Numeric:** Mean, Median, Min/Max, and Customizable Percentiles.
    * **Categorical:** Top (Mode), Frequency, and Unique Counts.
    * **Datetime:** Start/End ranges and Temporal Span.
* **Per-Column Analysis:** Breaks down data statistics for every individual feature rather than a global average.
* **Dynamic Data Loading:** Native support for `.npy` (DriftBench), `.csv`, and `.parquet` formats.
* **Quality Metrics:** Real-time calculation of missing value percentages and high-cardinality detection.
* **CI/CD Integrated:** Automated code quality enforcement using **Flake8** via GitHub Actions.

---

## 📂 Repository Structure

Following professional Python package best practices:

```text
dataprofiling/
├── .github/workflows/    # CI/CD Pipeline (Flake8 Linting)
├── data/                 # Dataset storage (.npy, .csv)
├── src/                  # Source Code
│   ├── __init__.py       # Package initializer
│   ├── dataprofiler.py   # Core profiling logic
│   └── dynamic.py        # Type-aware branching and loading
├── .gitignore            # Version control exclusions (venv, cache, junk)
├── README.md             # Project documentation
└── requirements.txt      # Project dependencies (NumPy, Pandas)