import os
from dynamic import Profiling
from visual import DataVisualizer
import pandas as pd
import numpy as np

baseline_file = "data/dataset_1.npy"
expected_file = "data/dataset_2.npy"

if not os.path.exists(expected_file) or not os.path.exists(baseline_file):
    print(f"Error: File not found!")
else:
    profiler = Profiling(baseline_file)
    report = profiler.data_stats(p_list=[25, 50, 75])

    # Save the JSON report
    profiler.save_report(report, "reports/stats_summary.json")
    print("Stats report saved.")

    profiler2 = Profiling(expected_file)
    report2 = profiler2.data_stats(p_list=[25, 50, 75])

    # Save the JSON report
    profiler2.save_report(report2, "reports/stats_summary2.json")
    print("Stats report saved.")

    # print("Available columns:", profiler.df.columns.tolist())

   # Visualization
    viz = DataVisualizer(df=profiler.df, stats=report)
    # viz.save_plots()
    # print("Plots saved to reports/plots/")

    # loop through all columns
    for col in profiler.df.columns:
        if pd.api.types.is_numeric_dtype(
                profiler.df[col]) and pd.api.types.is_numeric_dtype(
                profiler2.df[col]):

            # Columns Extraction
            expected_col = profiler2.df[col]
            actual_col = profiler.df[col]

            # PSI Calculation
            psi_score, exp_p, act_p = profiler.psi_cal(
                expected_col, actual_col, bins=10)

            # PSI Report
            if psi_score < 0.1:
                print(
                    f"PSI Score: {psi_score:.4f} - No significant change detected.")
            elif 0.1 <= psi_score < 0.25:
                print(
                    f"PSI Score: {psi_score:.4f} - Moderate change detected.")
            else:
                print(
                    f"PSI Score: {psi_score:.4f} - Significant change detected.")

            # Drift Plot
            viz.drift_plot(
                expected_col,
                actual_col,
                f"{col}_baseline",
                f"{col}_current",
                psi_score,
                exp_p,
                act_p)
