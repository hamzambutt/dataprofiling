import numpy as np
import pandas as pd
import os


class Profiling:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = self.load_data()

    def load_data(self):
        ext = os.path.splitext(self.file_path)[1].lower()

        if ext == '.csv':
            return pd.read_csv(self.file_path)
        elif ext == '.npy':
            data = np.load(self.file_path)
            return pd.DataFrame(
                data, columns=[
                    f"feature_{i}" for i in range(
                        data.shape[1])])
        elif ext == '.parquet':
            return pd.read_parquet(self.file_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

    def data_stats(self, p_list=[]):

        report = {}
        for col in self.df.columns:
            series = self.df[col]
            col_info = {
                    "dtype": str(series.dtype),
                    "missing_values": int(series.isnull().mean()*100),
                    "unique_values": series.nunique()
            }
            # Numeric Branch
            if pd.api.types.is_numeric_dtype(series):
                col_info.update({
                    "mean": series.mean(),
                    "median": series.median(),
                    "min": series.min(),
                    "max": series.max()
                })
                p_values = np.percentile(series.dropna(), p_list)
                for p, val in zip(p_list, p_values):
                    col_info[f"percentile_{p}"] = val
            # Categorical Branch
            elif pd.api.types.is_string_dtype(series):
                col_info.update ({
                    "top": series.mode()[0],
                    "freq": series.value_counts().iloc[0]
                })
                # DateTime Branch
            elif pd.api.types.is_datetime64_any_dtype(series):
                col_info.update({
                    "min": series.min(),
                    "max": series.max(),
                    "range": series.max() - series.min()
                })
            report[col] = col_info
        return report


profiler = Profiling('data/dataset_2.npy')
percentt = [25, 50, 75]
#print(profiler.data_stats(percentt))

import json

results = profiler.data_stats(percentt)
print(json.dumps(results, indent=4))