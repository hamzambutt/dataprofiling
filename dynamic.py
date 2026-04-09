import pandas as pd
import numpy as np
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
            return pd.DataFrame(data, columns=[f"feature_{i}" for i in range(data.shape[1])])
        elif ext == '.parquet':
            return pd.read_parquet(self.file_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

    def data_stats(self, p_list=[]):

        unique_values = np.unique(self.df)

        stats = {
            "metadata" :{
                "total_rows" : self.df.shape[0],
                "total_columns" : self.df.shape[1],

                "column_types": self.df.dtypes.astype(str).to_dict(),

                "unique_count" : len(unique_values),
                "null_percentage": (np.isnan(self.df).mean() * 100)
                },

            "summaray" : {

                "mean" : np.mean(self.df),
                "median" : np.median(self.df),
                "min" : np.min(self.df),
                "max" : np.max(self.df)
                }        
        }

        p_values = np.percentile(self.df, p_list)
        
        for p, val in zip(p_list, p_values):
            stats[f"percentile_{p}"] = val
        return stats

profiler = Profiling('dataset_2.npy')
percentt=[25,50,75]
print(profiler.data_stats(percentt))