import os
import json
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

    if "Billing Amount" in current.df.columns:
        current.df["Billing Amount"] = current.df["Billing Amount"] * 2.0

    report_current = current.data_stats()

    # Save the JSON report
    baseline.save_report(report_base, "reports/healthcare_baseline.json")
    current.save_report(report_current, "reports/healthcare_current.json")

    # Segment by Gender
    target_gender = "Female"
    baseline_segment = baseline.df[baseline.df["Gender"] == target_gender]
    current_segment = current.df[current.df["Gender"] == target_gender]

    base_max = baseline_segment["Billing Amount"].max()
    current_max = current_segment["Billing Amount"].max()

    if current_max > base_max:
        print(f"Standard Outlier Check: Breached for {target_gender}!")
    else:
        print(f"Standard Outlier Check: Passed for {target_gender}!")

    segement_psi, _, _ = baseline.psi_cal(
        baseline_col=baseline_segment["Billing Amount"],
        current_col=current_segment["Billing Amount"],
        bins=10,
    )

    print(f"Segmented PSI for {target_gender}: {segement_psi:.4f}")

    if segement_psi > 0.2:
        print(f"Segmented PSI Check: Breached for {target_gender}")
    else:
        print(f"Segmented PSI Check: Passed for {target_gender}")
