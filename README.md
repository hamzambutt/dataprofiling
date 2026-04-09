# Data Profiling & Drift Detection Toolkit 🚀

A high-performance Python utility designed to dynamically profile datasets and prepare them for drift detection. This project is developed as part of my backend development internship at **Biome**, focusing on automating data quality checks and statistical analysis for large-scale datasets.

---

## 🌟 Features

* **Dynamic Data Profiling:** Automatically handles `.npy` (DriftBench) and `.csv` formats.
* **Metadata Extraction:** Instantly identifies row/column counts and individual column data types.
* **Automated Statistics:** Computes mean, median, min, max, and customizable percentiles.
* **Quality Metrics:** Calculates null percentages and unique value counts to assess data "health" and corruption.
* **CI/CD Integrated:** Automated code quality enforcement using **Flake8** via GitHub Actions on every push.

---

## 📂 Repository Structure

Following professional Python package best practices:

```text
dataprofiling/
├── .github/workflows/    # CI/CD Pipeline (Flake8 Linting)
├── data/                 # Dataset storage (.npy files)
├── src/                  # Source Code
│   ├── __init__.py       # Package initializer
│   ├── dataprofiler.py   # Core profiling logic
│   └── dynamic.py        # Dynamic data loading and handling
├── .gitignore            # Version control exclusions
├── README.md             # Project documentation
└── requirements.txt      # Project dependencies (NumPy, Pandas, etc.)
