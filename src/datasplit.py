import pandas as pd
import os

os.makedirs("data", exist_ok=True)

try:
    df = pd.read_csv("data/train.csv")
    total_rows = len(df)
    print(f"Loaded train.csv with {total_rows} rows.")
except FileNotFoundError:
    print("Error: train.csv not found.")
    exit()

batch_size = total_rows // 3

batch_1 = df.iloc[:batch_size].copy()
batch_2 = df.iloc[batch_size: 2 * batch_size].copy()
batch_3 = df.iloc[2 * batch_size:].copy()

batch_3["MSSubClass"] = batch_3["MSSubClass"] * 1.5
batch_3["LotFrontage"] = "Wood"


batch_1.to_csv("data/batch_1.csv", index=False)
batch_2.to_csv("data/batch_2.csv", index=False)
batch_3.to_csv("data/batch_3_drifted.csv", index=False)
