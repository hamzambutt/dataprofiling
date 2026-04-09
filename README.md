Data Profiling & Drift Detection Toolkit
A high-performance Python utility designed to dynamically profile datasets and prepare them for drift detection. This project is developed as part of my backend development internship, focusing on automating data quality checks and statistical analysis.

🚀 Features
Dynamic Data Profiling: Automatically handles .npy (DriftBench) and .csv formats.

Metadata Extraction: Instantly identifies row/column counts and individual column data types.

Automated Statistics: Computes mean, median, min, max, and customizable percentiles.

Quality Metrics: Calculates null percentages and unique value counts to assess data "health."

CI/CD Integrated: Automated code quality enforcement using Flake8 via GitHub Actions.

📂 Repository Structure
Following standard Python package best practices:

Plaintext
dataprofiling/
├── .github/workflows/    # CI/CD Pipeline (Flake8 Linting)
├── data/                 # Dataset storage (.npy files)
├── src/                  # Source Code
│   ├── dataprofiler.py   # Core profiling logic
│   └── dynamic.py        # Dynamic data loading and handling
├── .gitignore            # Version control exclusions
├── README.md             # Project documentation
└── requirements.txt      # Project dependencies
🛠️ Installation & Setup
Clone the repository:

Bash
git clone https://github.com/hamzambutt/dataprofiling.git
cd dataprofiling
Set up a Virtual Environment:

Bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install Dependencies:

Bash
pip install -r requirements.txt
📊 Usage
To profile a dataset from the DriftBench suite, run the dynamic.py script:

Python
from src.dynamic import Profiling
import numpy as np

# Load data
data = np.load('data/dataset_1.npy')
profiler = Profiling(data)

# Generate profile with custom percentiles
stats = profiler.data_stats(p_list=[25, 50, 75])
print(stats)
🛡️ Code Quality (CI/CD)
This project maintains strict adherence to PEP 8 standards. Every push to the main branch triggers an automated Flake8 linting check.

Status: All builds are verified through GitHub Actions to ensure clean, readable, and maintainable code.
