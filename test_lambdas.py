import pandas as pd, numpy as np, sys, joblib
sys.path.insert(0, '.')
from src.tournament import build_match_features_for_prediction, build_team_stats, load_raw_data
from src.model import predict_goals

results_df, rankings = load_raw_data()
team_stats = build_team_stats(results_df, rankings)
ratings  = pd.read_csv('data/processed/attack_defense_ratings.csv')
squad_df = pd.read_csv('data/processed/squad_features.csv').set_index('team')
model_h  = joblib.load('models/model_home.pkl')
model_a  = joblib.load('models/model_away.pkl')
cols     = joblib.load('models/feature_cols.pkl')

matchups = [
    ('South Africa', 'Mexico'),
    ('Spain', 'France'),
    ('Brazil', 'Argentina'),
    ('Haiti', 'Norway'),
    ('South Africa', 'Spain'),
]

for h, a in matchups:
    feats = build_match_features_for_prediction(
        h, a, team_stats, results_df, True,
        ratings=ratings, squad_df=squad_df)
    lh, la = predict_goals(model_h, model_a, cols, feats)
    sv_h = feats['squad_value_home']
    sv_a = feats['squad_value_away']
    rk_h = feats['fifa_rank_home']
    rk_a = feats['fifa_rank_away']
    print(f"{h:20s} vs {a:15s}: lh={lh:.2f} la={la:.2f} | sq_h={sv_h:.0f} sq_a={sv_a:.0f} | rank_h={rk_h:.0f} rank_a={rk_a:.0f}")
