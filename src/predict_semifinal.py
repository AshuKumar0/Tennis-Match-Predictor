import pandas as pd
import os
import pickle

data_folder = "C:/project/data"
csv_files = [f for f in os.listdir(data_folder) if f.endswith(".csv")]
df_list = [pd.read_csv(os.path.join(data_folder, f)) for f in csv_files]
df = pd.concat(df_list, ignore_index=True)

model_path = "src/lr_model.pkl"
lr_model = pickle.load(open(model_path, "rb"))

player_stats = {}
for idx, row in df.iterrows():
    winner = row['winner_name']
    if winner not in player_stats:
        player_stats[winner] = {
            'rank': row['winner_rank'] if not pd.isna(row['winner_rank']) else 1500,
            'age': row['winner_age'] if not pd.isna(row['winner_age']) else 27,
            'height': row['winner_ht'] if not pd.isna(row['winner_ht']) else 183,
            'elo': row['winner_elo'] if 'winner_elo' in row and not pd.isna(row['winner_elo']) else 1500
        }
    loser = row['loser_name']
    if loser not in player_stats:
        player_stats[loser] = {
            'rank': row['loser_rank'] if not pd.isna(row['loser_rank']) else 1500,
            'age': row['loser_age'] if not pd.isna(row['loser_age']) else 27,
            'height': row['loser_ht'] if not pd.isna(row['loser_ht']) else 183,
            'elo': row['loser_elo'] if 'loser_elo' in row and not pd.isna(row['loser_elo']) else 1500
        }

sf = [ 
    ("Novak Djokovic", "Yannick Hanfmann"),
    ("Sebastian Korda", "Lorenzo Musetti")
]

winners = []
print("\n===== Semifinals =====")
for playerA, playerB in sf:
    statsA = player_stats.get(playerA)
    statsB = player_stats.get(playerB)
    if statsA is None or statsB is None:
        print(f"Stats missing for {playerA} or {playerB}, skipping match")
        continue
    rank_diff = statsA['rank'] - statsB['rank']
    age_diff = statsA['age'] - statsB['age']
    height_diff = statsA['height'] - statsB['height']
    elo_diff = statsA['elo'] - statsB['elo']
    surface_Clay = 0
    surface_Grass = 0
    surface_Hard = 1
    X_match = pd.DataFrame([[rank_diff, age_diff, height_diff, elo_diff,
                             surface_Clay, surface_Grass, surface_Hard]],
                           columns=['rank_diff','age_diff','height_diff','elo_diff',
                                    'surface_Clay','surface_Grass','surface_Hard'])
    prob = lr_model.predict_proba(X_match)[0][1]
    winner = playerA if prob >= 0.5 else playerB
    winners.append(winner)
    print(f"{playerA} vs {playerB} → Winner: {winner} (Prob {prob:.2f})")

