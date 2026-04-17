import pandas as pd
import os

os.makedirs("data", exist_ok=True)

try:
    df = pd.read_csv("data/SeoulBikeData.csv", encoding="latin1")
    total_rows = len(df)
    print(f"Loaded SeoulBikeData.csv with {total_rows} rows.")
except FileNotFoundError:
    print("Error: SeoulBikeData.csv not found.")
    exit()

print(total_rows)
# batch_size = total_rows // 5

"""batch_1 = df.iloc[:batch_size].copy()
batch_2 = df.iloc[batch_size: 2 * batch_size].copy()
batch_3 = df.iloc[2 * batch_size: 3 * batch_size].copy()
batch_4 = df.iloc[3 * batch_size: 4 * batch_size].copy()
batch_5 = df.iloc[4 * batch_size:].copy()


batch_1.to_csv("data/cycle_batch_1.csv", index=False)
batch_2.to_csv("data/cycle_batch_2.csv", index=False)
batch_3.to_csv("data/cycle_batch_3.csv", index=False)
batch_4.to_csv("data/cycle_batch_4.csv", index=False)
batch_5.to_csv("data/cycle_batch_5.csv", index=False)
"""
