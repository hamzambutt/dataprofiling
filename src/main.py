import os
from dynamic import Profiling
from visual import DataVisualizer
import pandas as pd
import json

with open("config.json", "r") as f:
    config = json.load(f)

baseline_file = config["datapaths"]["baseline"]
current_file = config["datapaths"]["current"]

# Thresholds
psi_ = config["threshold"]["psi_limit"]
chi_p = config["threshold"]["chi_p_value"]
psi_moderate = config["threshold"]["psi_moderate_limit"]
percentiles = config["threshold"]["percentile"]

if not os.path.exists(current_file) or not os.path.exists(baseline_file):
    print("Error: File not found!")
else:
    baseline = Profiling(baseline_file)
    report_base = baseline.data_stats(percentiles)

    # Save the JSON report
    baseline.save_report(report_base, "reports/base_file.json")
    print("Stats report saved.")

    current = Profiling(current_file)
    report_current = current.data_stats(percentiles)

    # Save the JSON report
    current.save_report(report_current, "reports/current_file.json")
    print("Stats report saved.")

    # print("Available columns:", profiler.df.columns.tolist())

    # Visualization
    viz = DataVisualizer(df=baseline.df, stats=report_base)
#    viz.save_plots()

common_cols = set(baseline.df.columns).intersection(set(current.df.columns))
drift_analysis = []
# loop through all columns
for col in common_cols:
    result = {"column": col}

    if pd.api.types.is_numeric_dtype(
        baseline.df[col]
    ) and pd.api.types.is_numeric_dtype(current.df[col]):

        # PSI Calculation
        psi_score, exp_p, act_p = baseline.psi_cal(
            current.df[col], baseline.df[col], bins=10
        )

        # PSI Report
        if psi_score < psi_:
            status = "Stable Distribution"
        elif psi_ <= psi_score < psi_moderate:
            status = "Moderate Distribution."
        else:
            status = "Significant Distribution."

        result.update(
            {
                "Method": "PSI",
                "PSI Score": round(psi_score, 4),
                "Status": status,
            }
        )

    else:
        chi_, p_value = baseline.chi_cal(current.df[col], baseline.df[col])
        if p_value < chi_p:
            status = "Significant change detected."
        else:
            status = "No significant change detected."

        result.update(
            {
                "Method": "Chi-Squared",
                "Chi-Squared Statistic": round(chi_, 4),
                "P-Value": round(p_value, 4),
                "Status": status,
            }
        )
    drift_analysis.append(result)

with open("reports/drift_analysis.json", "w") as f:
    json.dump(drift_analysis, f, indent=4)

    # Chi-Squared Report
#            chi_plt = viz.chi_plot(col, chi_current, chi_baseline)

# Drift Plot
"""            viz.drift_plot(
                current_col,
                baseline_col,
                f"{col}_baseline",
                f"{col}_current",
                psi_score,
                exp_p,
                act_p,
            )"""
