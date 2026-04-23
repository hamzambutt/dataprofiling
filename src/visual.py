import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os


class DataVisualizer:
    def __init__(self, df, stats):
        self.df = df
        self.stats = stats
        os.makedirs("reports/plots", exist_ok=True)

    def save_plots(self):
        for col_name in self.df.columns:
            series = self.df[col_name]

            # Numeric Branch
            if pd.api.types.is_numeric_dtype(series):
                self._plot_numeric(series, col_name)

            # Datetime Branch
            elif pd.api.types.is_datetime64_any_dtype(series):
                self._plot_datetime(series, col_name)

            # Categorical Branch
            elif pd.api.types.is_object_dtype(
                series
            ) or pd.api.types.is_string_dtype(series):
                self._plot_categorical(series, col_name)

    def _plot_numeric(self, series, col_name):
        data = series.dropna()
        if data.empty:
            return

        col_metrics = self.stats.get(col_name, {})

        fig, (ax_hist, ax_box) = plt.subplots(
            2, 1, figsize=(8, 8), gridspec_kw={"height_ratios": (0.8, 0.2)}
        )

        # Histogram
        ax_hist.hist(
            data, bins=30, color="skyblue", edgecolor="black", alpha=0.7
        )
        mean_val = col_metrics.get("mean")
        median_val = col_metrics.get("median")

        if mean_val is not None:
            ax_hist.axvline(
                mean_val,
                color="red",
                linestyle="--",
                label=f"Mean: {mean_val:.2f}",
            )
        if median_val is not None:
            ax_hist.axvline(
                median_val,
                color="green",
                linestyle=":",
                label=f"Median: {median_val:.2f}",
            )

        ax_hist.set_title(f"Distribution & Spread of {col_name}")
        ax_hist.set_ylabel("Frequency")
        ax_hist.legend()

        ax_box.boxplot(
            data,
            vert=False,
            patch_artist=True,
            boxprops=dict(facecolor="lightcoral", color="black"),
        )
        ax_box.set_xlabel("Value")
        ax_box.set_yticks([])

        plt.tight_layout()
        plt.savefig(f"reports/plots/{col_name}_numeric.png")
        plt.close()

    def _plot_categorical(self, series, col_name):
        """Generates a Bar Chart for the Top 10 categories."""
        top_10 = series.value_counts().head(10)
        if top_10.empty:
            return

        plt.figure(figsize=(10, 6))
        plt.bar(
            top_10.index.astype(str),
            top_10.values,
            color="mediumseagreen",
            edgecolor="black",
        )

        plt.title(f"Top 10 Categories in {col_name}")
        plt.xlabel("Categories")
        plt.ylabel("Frequency")
        plt.xticks(rotation=45, ha="right")

        plt.tight_layout()
        plt.savefig(f"reports/plots/{col_name}_categorical.png")
        plt.close()

    def _plot_datetime(self, series, col_name):
        """Generates a Time Series plot grouped by day."""
        data = series.dropna()
        if data.empty:
            return

        plt.figure(figsize=(10, 5))
        # Group by the date part and count
        counts = data.dt.date.value_counts().sort_index()

        plt.plot(
            counts.index,
            counts.values,
            color="teal",
            marker="o",
            linestyle="-",
        )

        plt.title(f"Record Counts Over Time: {col_name}")
        plt.xlabel("Date")
        plt.ylabel("Record Count")
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.savefig(f"reports/plots/{col_name}_time.png")
        plt.close()

    def drift_plot(self, baseline, current, col_1, col_2, psi, exp_p, act_p):
        """Generates a PSI plot comparing expected vs actual distributions."""
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.hist(
            baseline,
            bins=30,
            alpha=0.5,
            label="Expected",
            color="blue",
            density=True,
        )
        plt.hist(
            current,
            bins=30,
            alpha=0.5,
            label="Actual",
            color="orange",
            density=True,
        )

        # drift status based on PSI thresholds
        if psi < 0.1:
            status = "No significant change"
        elif 0.1 <= psi < 0.25:
            status = "Moderate change"
        else:
            status = "Significant change"

        plt.title(f"PSI Plot for {col_1} and {col_2}")
        plt.xlabel("Value")
        plt.ylabel("Density")
        plt.text(
            0.05,
            0.95,
            f"PSI: {psi:.4f}\nStatus: {status}",
            transform=plt.gca().transAxes,
        )
        plt.legend()

        # Bar plot of expected vs actual proportions
        plt.subplot(1, 2, 2)
        x = np.arange(len(exp_p))
        plt.bar(
            x - 0.2,
            exp_p,
            width=0.4,
            color="blue",
            alpha=0.7,
            label="Expected %",
        )
        plt.bar(
            x + 0.2,
            act_p,
            width=0.4,
            color="orange",
            alpha=0.7,
            label="Actual %",
        )

        plt.title("Bin-wise Proportions used in PSI")
        plt.xlabel("Bucket Number")
        plt.ylabel("Percentage of Data")
        plt.xticks(x, [f"Bin {i+1}" for i in range(len(exp_p))], rotation=45)
        plt.legend()

        plt.tight_layout()
        plt.savefig(f"reports/plots/{col_1}_psi.png")
        plt.close()

    def chi_plot(self, col_name, baseline_series, current_series):
        current_counts = current_series.value_counts()
        baseline_counts = baseline_series.value_counts()

        baseline_counts.index = baseline_counts.index.astype(str)
        current_counts.index = current_counts.index.astype(str)

        df_plot = pd.DataFrame(
            {"Expected": baseline_counts, "Actual": current_counts}
        ).fillna(0)

        df_plot.plot(kind="bar", figsize=(10, 6), color=["blue", "orange"])
        plt.title(f"Chi-Squared Distribution for {col_name}")
        plt.xlabel("Categories")
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"reports/plots/{col_name}_chi.png")
        plt.close()

    def plot_root_cause_evidence(
        self, base_df, curr_df, feature_col, target_col, country
    ):
        import matplotlib.pyplot as plt

        # Safely align and drop missing data for both columns simultaneously
        base_clean = base_df.dropna(subset=[feature_col, target_col])
        curr_clean = curr_df.dropna(subset=[feature_col, target_col])

        if base_clean.empty or curr_clean.empty:
            return

        plt.figure(figsize=(10, 6))

        # Plot Baseline (The Past)
        plt.scatter(
            base_clean[feature_col],
            base_clean[target_col],
            alpha=0.6,
            label="Baseline (Past)",
            color="#457b9d",
            edgecolor="black",
            s=80,
        )

        # Plot Current (The Anomaly)
        plt.scatter(
            curr_clean[feature_col],
            curr_clean[target_col],
            alpha=0.7,
            label="Current (Anomaly)",
            color="#e63946",
            edgecolor="black",
            s=80,
        )

        plt.title(
            f"Drift Impact: {feature_col} affected {target_col} in {country}",
            fontsize=14,
            fontweight="bold",
        )
        plt.xlabel(feature_col, fontsize=12)
        plt.ylabel(target_col, fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, linestyle="--", alpha=0.4)

        # Format safely for saving
        safe_feat = str(feature_col).replace(" ", "_").replace("/", "_")
        plt.tight_layout()
        plt.savefig(
            f"reports/plots/{country}_Evidence_{safe_feat}.png", dpi=300
        )
        plt.close()
