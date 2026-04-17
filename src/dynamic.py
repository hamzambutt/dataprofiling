import numpy as np
from scipy.stats import chi2_contingency
import pandas as pd
import os
import json


class Profiling:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = self.load_data()

    def load_data(self):
        ext = os.path.splitext(self.file_path)[1].lower()
        df = None

        if ext == ".csv":
            df = pd.read_csv(self.file_path)
        elif ext == ".npy":
            data = np.load(self.file_path)
            df = pd.DataFrame(data)
        elif ext == ".json":
            with open(self.file_path, "r") as f:
                data = json.load(f)
            df = pd.DataFrame(data)
        elif ext == ".parquet":
            df = pd.read_parquet(self.file_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

        # Dynamic Naming
        if isinstance(
            df.columns, pd.RangeIndex
        ) or pd.api.types.is_integer_dtype(df.columns):
            df.columns = [f"feature_{i}" for i in range(df.shape[1])]
        else:
            df.columns = df.columns.astype(str).str.strip()

        # 3. Clean up date formats, fix str
        for col in df.columns:
            if df[col].dtype == "object" or pd.api.types.is_string_dtype(
                df[col]
            ):
                try:
                    df[col] = pd.to_datetime(df[col], format="mixed")
                except (ValueError, TypeError, UserWarning):
                    pass

        return df

    def data_stats(self, p_list=[]):
        report = {}
        for col in self.df.columns:
            series = self.df[col]
            clean_series = series.dropna()

            col_info = {
                "dtype": str(series.dtype),
                # fix this
                "missing_values": float(series.isnull().sum()),
                "unique_values": int(series.nunique()),
            }

            if pd.api.types.is_numeric_dtype(series):
                if not clean_series.empty:
                    col_info.update(
                        {
                            "mean": float(clean_series.mean()),
                            "median": float(clean_series.median()),
                            "min": float(clean_series.min()),
                            "max": float(clean_series.max()),
                        }
                    )
                    p_values = np.percentile(clean_series, p_list)
                    for p, val in zip(p_list, p_values):
                        col_info[f"percentile_{p}"] = float(val)
                else:
                    col_info.update(
                        {
                            "mean": None,
                            "median": None,
                            "min": None,
                            "max": None,
                        }
                    )

            elif pd.api.types.is_object_dtype(
                series
            ) or pd.api.types.is_string_dtype(series):
                mode_res = series.mode()
                col_info.update(
                    {
                        "top": (
                            str(mode_res[0]) if not mode_res.empty else "N/A"
                        ),
                        "freq": (
                            int(series.value_counts().iloc[0])
                            if not series.empty
                            else 0
                        ),
                    }
                )

            elif pd.api.types.is_datetime64_any_dtype(series):
                if not clean_series.empty:
                    col_info.update(
                        {
                            "min": str(clean_series.min()),
                            "max": str(clean_series.max()),
                            "range": str(
                                clean_series.max() - clean_series.min()
                            ),
                        }
                    )
                else:
                    col_info.update({"min": None, "max": None, "range": None})

            report[col] = col_info
        return report

    def save_report(self, report, output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=4)

    def psi_cal(self, baseline_col, current_col, bins):

        expected_col = baseline_col.dropna()
        actual_col = current_col.dropna()
        breakpoints = np.unique(
            np.percentile(expected_col, np.linspace(0, 100, bins + 1))
        )

        expected_counts, _ = np.histogram(
            expected_col.dropna(), bins=breakpoints
        )
        actual_counts, _ = np.histogram(actual_col.dropna(), bins=breakpoints)

        expected_percents = expected_counts / len(expected_col.dropna())
        actual_percents = actual_counts / len(actual_col.dropna())

        # 1e-10 to prevent division by zero and log of zero
        expected_percents += 1e-10
        actual_percents += 1e-10

        psi_vals = np.sum(
            (expected_percents - actual_percents)
            * np.log(expected_percents / actual_percents)
        )

        return psi_vals, expected_percents, actual_percents

    def chi_cal(self, baseline_col, current_col):

        expected_counts = baseline_col.value_counts(normalize=False)
        actual_counts = current_col.value_counts(normalize=False)

        expected_counts, actual_counts = expected_counts.align(
            actual_counts, fill_value=1e-5
        )

        chi2_stat, p_value, _, _ = chi2_contingency(
            [expected_counts, actual_counts]
        )

        return chi2_stat, p_value
