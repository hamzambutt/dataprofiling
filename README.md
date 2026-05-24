# Data Profiling & Dual-Engine Analytics Pipeline 🚀

[![Flake8 Linting](https://img.shields.io/badge/Linting-Flake8-blue)](#)
[![PySpark Tests](https://img.shields.io/badge/PySpark_Tests-Passing-success?logo=apachespark)](#)
[![Ibis Tests](https://img.shields.io/badge/Ibis_Tests-Passing-success?logo=python)](#)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue)](#)

A professional-grade Python toolkit designed for **Per-Column Dynamic Profiling** and **Large-Scale Data Analytics**. 

This project is split into two core modules: a dynamic data profiler that adapts statistical analysis based on data types, and a side-by-side implementation of industry-standard Big Data processing (PySpark) alongside modern, lightweight analytical frameworks (Ibis + DuckDB).

Developed as part of the Backend Development Internship at **Biome**.

---

## 🌟 Key Features

### 1. Dynamic Data Profiler
* **Type-Aware Branching Logic:** Automatically detects and applies specific profiling metrics for:
    * **Numeric:** Mean, Median, Min/Max, and Customizable Percentiles.
    * **Categorical:** Top (Mode), Frequency, and Unique Counts.
    * **Datetime:** Start/End ranges and Temporal Span.
* **Dynamic Data Loading:** Native support for `.npy` (DriftBench), `.csv`, and `.parquet` formats.
* **Quality Metrics:** Real-time calculation of missing value percentages and high-cardinality detection.

### 2. Dual-Engine Analytics (Cruise Ship Dataset)
* **PySpark (Distributed Processing):** Enterprise-grade analytics pipeline built on PySpark (v3.5+) for scalable, distributed data processing.
* **Ibis + DuckDB (Local Execution):** Identical analytical logic translated into the Ibis framework, executing in milliseconds via an in-memory DuckDB engine.
* **Extracted Insights:** Calculates fleet demographics, Crew-to-Passenger service ratios, tonnage rankings, and statistical correlations between ship capacity and physical size.

### 3. Automated CI/CD & Testing
* **Unified Assertions:** Both data engines are rigorously tested using `pandas.testing.assert_frame_equal`, ensuring strict validation of schemas, floating-point precision, and data integrity.
* **Parallel GitHub Actions:** Fully automated workflows that run Flake8 code quality enforcement, an isolated Java 17 environment for PySpark tests, and a pure-Python environment for Ibis tests concurrently on every push.

---

## 📂 Repository Structure

```text
dataprofiling/
├── .github/workflows/    # CI/CD Pipelines
│   ├── flake8-lint.yml   # Code quality enforcement
│   ├── ibis-tests.yml    # Automated DuckDB testing
│   └── pyspark-tests.yml # Automated PySpark testing (Java 17)
├── data/                 # Dataset storage (.npy, .csv)
├── src/                  # Core Profiling Logic
│   ├── __init__.py       
│   ├── dataprofiler.py   # Per-column statistical analysis
│   └── dynamic.py        # Type-aware branching and loading
├── Ibis/                 # Modern Analytics Pipeline
│   ├── analytic_ibis.py  # Ibis/DuckDB data transformations
│   └── test_ibis.py      # Ibis unit tests 
├── Spark/                # Distributed Analytics Pipeline
│   ├── spark.py          # PySpark DataFrame transformations
│   └── test_spark.py     # PySpark unit tests 
├── README.md             # Project documentation
└── requirements.txt      # Project dependencies
