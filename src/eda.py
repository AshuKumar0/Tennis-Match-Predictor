import pandas as pd
import glob

path = "data/*.csv"
files = glob.glob(path)

print("Number of files found:", len(files))

if len(files) == 0:
    raise FileNotFoundError("No CSV files found in data folder")

df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

print(df.shape)
print(df.head())

print(df.columns.tolist())

print(df.isnull().sum().sort_values(ascending=False).head(15))

print(df['surface'].value_counts())

better_rank_wins = (df['winner_rank'] < df['loser_rank']).mean()
print(f"Better ranked player wins: {better_rank_wins:.2%}")
