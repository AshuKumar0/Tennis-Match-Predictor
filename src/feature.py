import pandas as pd
import glob
from math import pow

INITIAL_ELO = 1500
K = 32

def expected_score(player_elo, opponent_elo):
    return 1 / (1 + pow(10, (opponent_elo - player_elo) / 400))

files = glob.glob("data/*.csv")
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

df = df.dropna(subset=[
    'tourney_date',
    'winner_name', 'loser_name',
    'winner_rank', 'loser_rank',
    'winner_age', 'loser_age',
    'winner_ht', 'loser_ht',
    'surface'
])

df['tourney_date'] = pd.to_datetime(df['tourney_date'], format='%Y%m%d')
df = df.sort_values('tourney_date').reset_index(drop=True)

elo = {}
rows = []

for _, row in df.iterrows():

    winner = row['winner_name']
    loser = row['loser_name']

    if winner not in elo:
        elo[winner] = INITIAL_ELO
    if loser not in elo:
        elo[loser] = INITIAL_ELO

    winner_elo = elo[winner]
    loser_elo = elo[loser]

    rank_diff = row['winner_rank'] - row['loser_rank']
    age_diff = row['winner_age'] - row['loser_age']
    height_diff = row['winner_ht'] - row['loser_ht']
    elo_diff = winner_elo - loser_elo

    surface = row['surface']

    rows.append({
        'rank_diff': rank_diff,
        'age_diff': age_diff,
        'height_diff': height_diff,
        'elo_diff': elo_diff,
        'surface': surface,
        'target': 1
    })

    rows.append({
        'rank_diff': -rank_diff,
        'age_diff': -age_diff,
        'height_diff': -height_diff,
        'elo_diff': -elo_diff,
        'surface': surface,
        'target': 0
    })

    expected_w = expected_score(winner_elo, loser_elo)
    expected_l = expected_score(loser_elo, winner_elo)

    elo[winner] = winner_elo + K * (1 - expected_w)
    elo[loser] = loser_elo + K * (0 - expected_l)

ml_df = pd.DataFrame(rows)

ml_df = pd.get_dummies(ml_df, columns=['surface'], drop_first=True)

ml_df.to_csv("data/ml_dataset_elo.csv", index=False)

print("Feature engineering completed")
print("Dataset shape:", ml_df.shape)
print(ml_df.head())
