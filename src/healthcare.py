import os
import json
import pandas as pd
from dynamic import Profiling

with open("config.json", "r") as f:
    config = json.load(f)

baseline_file = config["datapaths"]["baseline"]
current_file = config["datapaths"]["current"]

if not os.path.exists(current_file) or not os.path.exists(baseline_file):
    print("Error: File not found!")
else:
    baseline = Profiling(baseline_file)
    report_base = baseline.data_stats()

    current = Profiling(current_file)

    # injection for testing
    if "Billing Amount" in current.df.columns:
        current.df["Billing Amount"] = current.df["Billing Amount"] * 2.0

    report_current = current.data_stats()

    # Save the JSON report
    baseline.save_report(report_base, "reports/healthcare_baseline.json")

    # Segment by Gender
    target_segment = "Medical Condition"
    unique_segments = baseline.df[target_segment].dropna().unique()

    for condition in unique_segments:

        baseline_segment = baseline.df[
            baseline.df[target_segment] == condition
        ]
        current_segment = current.df[current.df[target_segment] == condition]

        for col in baseline_segment.columns:

            if col == target_segment:
                continue  # skips the column

            if pd.api.types.is_numeric_dtype(baseline.df[col]):
                segement_psi, _, _ = baseline.psi_cal(
                    baseline_segment[col],
                    current_segment[col],
                    bins=10,
                )
                if segement_psi > config["threshold"]["psi_limit"]:
                    print(
                        f"Critical drift detected in column {col}"
                        f" for segment {condition}"
                    )
                elif segement_psi > config["threshold"]["psi_moderate_limit"]:
                    print(
                        f"Moderate drift detected in column {col}"
                        f" for segment {condition}"
                    )
                else:
                    print(
                        f"No significant drift detected in column {col}"
                        f" for segment {condition}"
                    )

            else:
                chi_stat, p_value = baseline.chi_cal(
                    baseline_segment[col], current_segment[col]
                )

                if p_value < config["threshold"]["chi_p_value"]:
                    print(
                        f"Critical drift detected in column {col}"
                        f" for segment {condition}"
                    )
