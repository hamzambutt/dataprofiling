import pandas as pd
import numpy as np


class Profiling:
    def __init__(self,df):
        self.df= np.array(df)

    def data_stats(self, p=45):
        stats= {
        "mean" : np.mean(self.df),
       "median" : np.median(self.df),
        "min" : np.min(self.df),
        "max" : np.max(self.df),
        "percentile" : np.percentile(self.df,p)
        }
        return stats

data = np.load('dataset_1.npy')
profiler = Profiling(data)
print(profiler.data_stats())
