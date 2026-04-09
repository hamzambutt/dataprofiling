import pandas as pd
import numpy as np


class Profiling:
    def __init__(self,df):
        self.df= np.array(df)

    def data_stats(self, p_list=[]):

        unique_values = np.unique(self.df)

        stats = {
            "metadata" :{
                "total_rows" : self.df.shape[0],
                "total_columns" : self.df.shape[1],

                "datatype" : self.df.dtype,

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

data = np.load('dataset_1.npy')
profiler = Profiling(data)
percentt=[25,50,75]
print(profiler.data_stats(percentt))