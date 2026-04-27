import pandas as pd
import os

os.makedirs("data", exist_ok=True)

try:
    # Load the data safely
    df = pd.read_csv("data/LifeExpectancy.csv", encoding="latin1")
    print(f"Loaded LifeExpectancy.csv with {len(df)} rows.")
except FileNotFoundError:
    print("Error: data/LifeExpectancy.csv not found.")
    exit()

# ==========================================
# 🕒 EVEN CHRONOLOGICAL SPLIT (8 Years vs 8 Years)
# ==========================================
if "Year" in df.columns:
    print("Splitting timeline evenly in half...")

    # Baseline = The Past 8 Years (2000 through 2007)
    batch_1 = df[df["Year"] <= 2007].copy()

    # Current = The Recent 8 Years (2008 through 2015)
    batch_2 = df[df["Year"] >= 2008].copy()

    print(
        f"Baseline (Past):   {batch_1['Year'].min()} "
        f"to {batch_1['Year'].max()}"
    )
    print(
        f"Current (Anomaly): {batch_2['Year'].min()} "
        f"to {batch_2['Year'].max()}"
    )

else:
    # Fallback just in case the Year column is missing
    print("Warning: No 'Year' column found. Falling back to 50/50 row split.")
    batch_size = len(df) // 2
    batch_1 = df.iloc[:batch_size].copy()
    batch_2 = df.iloc[batch_size:].copy()

print(f"\nBaseline Size: {len(batch_1)} rows")
print(f"Current Size:  {len(batch_2)} rows")

# Save the new temporal batches
batch_1.to_csv("data/life_baseline.csv", index=False)
batch_2.to_csv("data/life_current.csv", index=False)
