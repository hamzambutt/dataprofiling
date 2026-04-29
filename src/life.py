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
current = Profiling(current_file)

red_cols = [
    "infant deaths",
    "under-five deaths",
    "Year",
    "thinness  1-19 years",
    "thinness 5-9 years",
    "Schooling",
]

baseline.df = baseline.df.drop(columns=red_cols, errors="ignore")
current.df = current.df.drop(columns=red_cols, errors="ignore")

report_base = baseline.data_stats()

# drift injection for Afghanistan
if "Country" in current.df.columns:
    mask = current.df["Country"] == "Afghanistan"
    if "Hepatitis B" in current.df.columns:
        current.df["Hepatitis B"] = current.df["Hepatitis B"].astype(float)
        current.df.loc[mask, "Hepatitis B"] *= 1.60
    if "BMI" in current.df.columns:
        current.df["BMI"] = current.df["BMI"].astype(float)
        current.df.loc[mask, "BMI"] *= 2.60
    if "Polio" in current.df.columns:
        current.df["Polio"] = current.df["Polio"].astype(float)
        current.df.loc[mask, "Polio"] *= 5.60


report_current = current.data_stats()
baseline.save_report(report_base, "reports/life_baseline.json")

location_segment = "Country"
unique_country = baseline.df[location_segment].dropna().unique()

psi_limit = config["threshold"]["psi_limit"]
viz = DataVisualizer(current.df, report_base)

target_country = "Afghanistan"
injected_features = ["Hepatitis B", "BMI", "Polio"]

base_afg = baseline.df[baseline.df["Country"] == target_country]
current_afg = current.df[current.df["Country"] == target_country]

if not base_afg.empty and not current_afg.empty:
    drift_data = []
    drift_num = {}
    for feature in injected_features:
        if feature in base_afg.columns and feature in current_afg.columns:
            feature_psi, exp_p, act_p = baseline.psi_cal(
                base_afg[feature].dropna(),
                current_afg[feature].dropna(),
                bins=10,
            )
            ks_stat, ks_p_value = baseline.ks_cal(
                base_afg[feature], current_afg[feature]
            )
            em_stat = baseline.em_cal(base_afg[feature], current_afg[feature])
            js_stat = baseline.js_cal(
                base_afg[feature], current_afg[feature], bins=10
            )

            print(
                f"{feature} (PSI: {feature_psi:.4f}, KS: {ks_p_value:.4f},"
                f"JS: {js_stat:.4f}, EM: {em_stat:.4f})"
            )

            drift_num[feature] = {
                "PSI_Score": round(float(feature_psi), 4),
                "KS_P_Value": round(float(ks_p_value), 4),
                "EMD_Score": round(float(em_stat), 4),
                "JS_Score": round(float(js_stat), 4),
                "Old_Average": round(float(base_afg[feature].mean()), 2),
                "New_Average": round(float(current_afg[feature].mean()), 2),
            }

            drift_data.append(
                {
                    "Feature": feature,
                    "Dataset": "Baseline",
                    "Value": base_afg[feature].mean(),
                }
            )
            drift_data.append(
                {
                    "Feature": feature,
                    "Dataset": "Current",
                    "Value": current_afg[feature].mean(),
                }
            )
    report_path = f"reports/{target_country}_drift_alarms.json"
    with open(report_path, "w") as f:
        json.dump(drift_num, f, indent=4)
    drift_df = pd.DataFrame(drift_data)

    # viz.plot_afghanistan_drift(drift_df, target_country)


# Country with the highest average Adult Mortality and its associated features
target_col = "Adult Mortality"
base_df = baseline.df

if "Country" in base_df.columns and target_col in base_df.columns:

    country_means = (
        base_df.groupby("Country").mean(numeric_only=True).reset_index()
    )

    country_status_map = {}
    if "Status" in base_df.columns:
        for index, row in base_df.iterrows():
            country_name = row["Country"]
            status = row["Status"]
            if pd.notna(status):
                country_status_map[country_name] = status
    # Identify numeric features to analyze
    features = []
    for col in country_means.columns:
        if col != "Country" and col != target_col and col != "Year":
            features.append(col)

    if len(features) > 0:

        plot_data = []
        for feature in features:
            sorted_df = country_means.sort_values(by=feature, ascending=False)
            winning_row = sorted_df.iloc[0]
            country_name = winning_row["Country"]

            plot_data.append(
                {
                    "Feature": feature,
                    "Highest Country": country_name,
                    "Mortality": winning_row[target_col],
                    "Status": country_status_map.get(country_name, "Unknown"),
                }
            )

        plot_df = pd.DataFrame(plot_data).sort_values(
            by="Mortality", ascending=False
        )

        viz.plot_feature_extremes(plot_df, target_col)

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
        if col == location_segment:
            continue  # Skip the location column itself
        if col == target_segment:
            continue  # Skip the target column itself

        if pd.api.types.is_numeric_dtype(
            baseline_segment[col]
        ) and pd.api.types.is_numeric_dtype(current_segment[col]):
            feature_psi, exp_p, act_p = baseline.psi_cal(
                baseline_segment[col], current_segment[col], bins=10
            )

            if feature_psi > psi_limit:
                base_mean = baseline_segment[col].mean()
                curr_mean = current_segment[col].mean()
                print(
                    f"{col} (Base: {base_mean:.2f} -> Curr: {curr_mean:.2f})"
                )

"""
            # Format name safely for the file saving
            safe_col_name = col.replace(" ", "_")
            plot_id = f"{country}_{safe_col_name}"

            country_status = "Unknown"
            if (
                "Status" in baseline_segment.columns
                and not baseline_segment["Status"].empty
            ):
                country_status = str(baseline_segment["Status"].iloc[0])

                viz.plot_root_cause_evidence(
                    base_df=baseline_segment,
                    curr_df=current_segment,
                    feature_col=col,
                    target_col=target_segment,
                    country=country,
                    status=country_status,
                )
        elif pd.api.types.is_object_dtype(
            baseline_segment[col]
        ) and pd.api.types.is_object_dtype(current_segment[col]):

            chi_stat, p_value = baseline.chi_cal(
                baseline_segment[col], current_segment[col]
            )

            if p_value < config["threshold"]["chi_p_value"]:
                print(f"{col} (Chi p-value: {p_value:.4f})")
                viz.chi_plot(
                    plot_id, baseline_segment[col], current_segment[col]
                )
                print(
                    f"Categorical Evidence saved: reports/plots/"
                    f"{plot_id}_chi.png\n"
                )


              # GENERATE THE DUAL-CHART DRIFT PLOT
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
                )
"""
