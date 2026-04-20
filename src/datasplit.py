import pandas as pd
import os

os.makedirs("data", exist_ok=True)

try:
    df = pd.read_csv("data/healthcare_dataset.csv", encoding="latin1")
    total_rows = len(df)
    print(f"Loaded healthcare_dataset.csv with {total_rows} rows.")
except FileNotFoundError:
    print("Error: healthcare_dataset.csv not found.")
    exit()

batch_size = total_rows // 2

batch_1 = df.iloc[:batch_size].copy()
batch_2 = df.iloc[batch_size:].copy()


batch_1.to_csv("data/healthcare_baseline.csv", index=False)
batch_2.to_csv("data/healthcare_current.csv", index=False)
