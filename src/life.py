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

if "Country" in current.df.columns:
    mask = current.df["Country"] == "Brazil"
    if "infant deaths" in current.df.columns:
        current.df["infant deaths"] = current.df["infant deaths"].astype(float)
        current.df.loc[mask, "infant deaths"] *= 1.60
    if "under-five deaths" in current.df.columns:
        current.df["under-five deaths"] = current.df[
            "under-five deaths"
        ].astype(float)
        current.df.loc[mask, "under-five deaths"] *= 1.60

report_current = current.data_stats()
baseline.save_report(report_base, "reports/life_baseline.json")

location_segment = "Country"
mortality_segment = ["Adult Mortality", "infant deaths", "under-five deaths"]
unique_country = baseline.df[location_segment].dropna().unique()

for country in unique_country:
    baseline_segment = baseline.df[baseline.df[location_segment] == country]
    current_segment = current.df[current.df[location_segment] == country]

    if baseline_segment.empty or current_segment.empty:
        print(
            f"Warning: No data for country {country} in one of the datasets."
        )
        continue

    for col in mortality_segment:
        if col in baseline.df.columns and pd.api.types.is_numeric_dtype(
            baseline.df[col]
        ):
            b_mean = baseline_segment[col].mean()
            c_mean = current_segment[col].mean()

            b_med = baseline_segment[col].median()
            c_med = current_segment[col].median()

            b_min = baseline_segment[col].min()
            c_min = current_segment[col].min()

            b_max = baseline_segment[col].max()
            c_max = current_segment[col].max()

            segement_psi, exp_p, act_p = baseline.psi_cal(
                baseline_segment[col], current_segment[col], bins=10
            )

            if segement_psi > config["threshold"]["psi_limit"]:
                print(f"Critical drift | {country} | {col}")
                print(f"  Mean:   Base {b_mean:.2f} | Curr {c_mean:.2f}")
                print(f"  Median: Base {b_med:.2f} | Curr {c_med:.2f}")
                print(f"  Min:    Base {b_min:.2f} | Curr {c_min:.2f}")
                print(f"  Max:    Base {b_max:.2f} | Curr {c_max:.2f}\n")
            elif segement_psi > config["threshold"]["psi_moderate_limit"]:
                print(f"Moderate drift | {country} | {col}")
                print(f"  Mean:   Base {b_mean:.2f} | Curr {c_mean:.2f}")
                print(f"  Median: Base {b_med:.2f} | Curr {c_med:.2f}")
                print(f"  Min:    Base {b_min:.2f} | Curr {c_min:.2f}")
                print(f"  Max:    Base {b_max:.2f} | Curr {c_max:.2f}\n")
            is_drifted = False

            if segement_psi > config["threshold"]["psi_limit"]:
                print(f"Critical drift | {country} | {col}")
                is_drifted = True
            elif segement_psi > config["threshold"]["psi_moderate_limit"]:
                print(f"Moderate drift | {country} | {col}")
                is_drifted = True

            if is_drifted:
                print(f"  Mean:   Base {b_mean:.2f} | Curr {c_mean:.2f}")
                print(f"  Median: Base {b_med:.2f} | Curr {c_med:.2f}")
                print(f"  Min:    Base {b_min:.2f} | Curr {c_min:.2f}")
                print(f"  Max:    Base {b_max:.2f} | Curr {c_max:.2f}")

                # Format name safely for the file saving
                safe_col_name = col.replace(" ", "_")
                plot_id = f"{country}_{safe_col_name}"

                # GENERATE THE DUAL-CHART DRIFT PLOT
                viz.drift_plot(
                    baseline=baseline_segment[col].dropna(),
                    current=current_segment[col].dropna(),
                    col_1=plot_id,
                    col_2="Current",
                    psi=segement_psi,
                    exp_p=exp_p,
                    act_p=act_p,
                )
                print(
                    f"Visual evidence saved: reports/plots/{plot_id}_psi.png\n"
                )
