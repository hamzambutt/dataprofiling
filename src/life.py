import os
import json
import pandas as pd
from dynamic import Profiling
from visual import DataVisualizer

with open("config.json", "r") as f:
    config = json.load(f)

baseline_file = config["datapaths"]["baseline"]
current_file = config["datapaths"]["current"]

if not os.path.exists(current_file) or not os.path.exists(baseline_file):
    raise FileNotFoundError("Baseline or current file not found.")

baseline = Profiling(baseline_file)
report_base = baseline.data_stats()
current = Profiling(current_file)
viz = DataVisualizer(current.df, report_base)

red_cols = [
    "infant deaths",
    "under-five deaths",
    "Year",
    "thinness  1-19 years",
    "thinness 5-9 years",
    "Schooling",
]

if "Country" in current.df.columns:
    mask = current.df["Country"] == "Afghanistan"
    if "Alcohol" in current.df.columns:
        current.df["Alcohol"] = current.df["Alcohol"].astype(float)
        current.df.loc[mask, "Alcohol"] *= 1.60

report_current = current.data_stats()
baseline.save_report(report_base, "reports/life_baseline.json")

location_segment = "Country"
unique_country = baseline.df[location_segment].dropna().unique()

psi_limit = config["threshold"]["psi_limit"]

for country in unique_country:
    baseline_segment = baseline.df[baseline.df[location_segment] == country]
    current_segment = current.df[current.df[location_segment] == country]

    if baseline_segment.empty or current_segment.empty:
        print(
            f"Warning: No data for country {country} in one of the datasets."
        )
        continue
    target_segment = "Adult Mortality"
    for col in baseline_segment.columns:
        if col in red_cols:
            continue  # skip red-flagged
        if col == target_segment:
            continue  # skip the target column
        if not pd.api.types.is_numeric_dtype(current.df[col]):
            continue  # skip non-numeric column

        feature_psi, exp_p, act_p = baseline.psi_cal(
            baseline_segment[col], current_segment[col], bins=10
        )

        if feature_psi > psi_limit:
            base_mean = baseline_segment[col].mean()
            curr_mean = current_segment[col].mean()
            print(f"{col} (Base: {base_mean:.2f} -> Curr: {curr_mean:.2f})")

            # Format name safely for the file saving
            safe_col_name = col.replace(" ", "_")
            plot_id = f"{country}_{safe_col_name}"

            viz.plot_root_cause_evidence(
                base_df=baseline_segment,
                curr_df=current_segment,
                feature_col=col,
                target_col=target_segment,
                country=country,
            )

"""                # GENERATE THE DUAL-CHART DRIFT PLOT
                viz.drift_plot(
                    baseline=baseline_segment[col].dropna(),
                    current=current_segment[col].dropna(),
                    col_1=plot_id,
                    col_2="Current",
                    psi=feature_psi,
                    exp_p=exp_p,
                    act_p=act_p,
                )
                print(
                    f"Visual evidence saved: reports/plots/{plot_id}_psi.png\n"
                )"""
