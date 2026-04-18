"""
tournament.py - Simulación completa del Mundial 2026.

Formato 2026:
  - 48 equipos, 12 grupos de 4
  - Fase de grupos: todos contra todos (3 partidos por equipo)
  - Clasifican: 2 primeros de cada grupo + 8 mejores terceros = 32 equipos
  - Ronda 1/32 -> 1/16 -> Cuartos -> Semis -> Final
  - Empates en eliminatorias -> penaltis (50/50)
"""

import numpy as np
import pandas as pd
from typing import Optional

from src.utils import HOST_NATIONS, logger, get_groups, print_bracket, save_results
from src.model import predict_goals, load_trained_models
from src.data_preparation import build_team_stats, load_raw_data, compute_h2h


def build_match_features_for_prediction(home, away, team_stats, results_df, is_neutral=False):
    def get_stats(team):
        if team in team_stats.index:
            s = team_stats.loc[team]
            return {
                "fifa_points": float(s.get("fifa_points", 1200)),
                "fifa_rank":   float(s.get("fifa_rank", 50)),
                "avg_scored":  float(s.get("avg_goals_scored", 1.2)),
                "avg_conceded":float(s.get("avg_goals_conceded", 1.2)),
                "win_rate":    float(s.get("win_rate", 0.33)),
            }
        return {"fifa_points": 1200, "fifa_rank": 50,
                "avg_scored": 1.2, "avg_conceded": 1.2, "win_rate": 0.33}

    h = get_stats(home)
    a = get_stats(away)
    h2h = compute_h2h(results_df, home, away)

    return {
        "fifa_points_home":    h["fifa_points"],
        "fifa_points_away":    a["fifa_points"],
        "fifa_rank_home":      h["fifa_rank"],
        "fifa_rank_away":      a["fifa_rank"],
        "rank_diff":           h["fifa_rank"] - a["fifa_rank"],
        "points_ratio":        h["fifa_points"] / max(a["fifa_points"], 1),
        "avg_scored_home":     h["avg_scored"],
        "avg_scored_away":     a["avg_scored"],
        "avg_conceded_home":   h["avg_conceded"],
        "avg_conceded_away":   a["avg_conceded"],
        "win_rate_home":       h["win_rate"],
        "win_rate_away":       a["win_rate"],
        "is_neutral":          int(is_neutral),
        "home_is_host_nation": int(home in HOST_NATIONS),
        "away_is_host_nation": int(away in HOST_NATIONS),
    }


def simulate_match(home, away, model_home, model_away, feature_cols,
                   team_stats, results_df, is_neutral=False, allow_draw=True):
    features = build_match_features_for_prediction(
        home, away, team_stats, results_df, is_neutral)
    lambda_home, lambda_away = predict_goals(
        model_home, model_away, feature_cols, features)

    hg = int(np.random.poisson(lambda_home))
    ag = int(np.random.poisson(lambda_away))

    went_to_penalties = False
    if not allow_draw and hg == ag:
        winner = home if np.random.random() < 0.5 else away
        went_to_penalties = True
    elif hg > ag:
        winner = home
    elif ag > hg:
        winner = away
    else:
        winner = "Draw"

    return {
        "home": home, "away": away,
        "home_goals": hg, "away_goals": ag,
        "winner": winner,
        "went_to_penalties": went_to_penalties,
        "lambda_home": round(lambda_home, 3),
        "lambda_away": round(lambda_away, 3),
    }


def simulate_group_stage(groups, model_home, model_away, feature_cols, team_stats, results_df):
    logger.info("\nSimulando fase de grupos...")
    group_results = {}

    for group_name, teams in groups.items():
        standings = {t: {"pts": 0, "gf": 0, "gc": 0, "gd": 0} for t in teams}
        match_log = []

        for i in range(len(teams)):
            for j in range(i + 1, len(teams)):
                home, away = teams[i], teams[j]
                result = simulate_match(
                    home, away, model_home, model_away, feature_cols,
                    team_stats, results_df, is_neutral=True, allow_draw=True)

                hg, ag = result["home_goals"], result["away_goals"]
                standings[home]["gf"] += hg; standings[home]["gc"] += ag
                standings[home]["gd"] += hg - ag
                standings[away]["gf"] += ag; standings[away]["gc"] += hg
                standings[away]["gd"] += ag - hg

                if hg > ag:
                    standings[home]["pts"] += 3
                elif ag > hg:
                    standings[away]["pts"] += 3
                else:
                    standings[home]["pts"] += 1
                    standings[away]["pts"]  += 1

                match_log.append({"group": group_name, **result})

        table = (pd.DataFrame(standings).T
                   .reset_index()
                   .rename(columns={"index": "team"})
                   .sort_values(["pts", "gd", "gf"], ascending=False)
                   .reset_index(drop=True))
        table["position"] = range(1, len(table) + 1)
        table["group"] = group_name

        group_results[group_name] = {
            "table": table,
            "matches": pd.DataFrame(match_log),
        }

    return group_results


def get_qualified_teams(group_results):
    qualified = []
    third_place = []

    for gname, data in group_results.items():
        table = data["table"]
        qualified.append(table.iloc[0]["team"])
        qualified.append(table.iloc[1]["team"])
        if len(table) > 2:
            row = table.iloc[2].to_dict()
            row["group"] = gname
            third_place.append(row)

    if third_place:
        thirds = (pd.DataFrame(third_place)
                    .sort_values(["pts", "gd", "gf"], ascending=False)
                    .head(8))
        qualified.extend(thirds["team"].tolist())

    logger.info(f"Equipos clasificados: {len(qualified)}")
    return qualified[:32]


def simulate_knockout_stage(qualified, model_home, model_away, feature_cols, team_stats, results_df):
    logger.info("\nSimulando fase eliminatoria...")
    bracket = {}
    current_round = qualified.copy()
    stage_names = ["Round of 32", "Round of 16", "Quarter-finals", "Semi-finals", "Final"]

    for stage in stage_names:
        if len(current_round) < 2:
            break
        logger.info(f"  -> {stage} ({len(current_round)} equipos)")
        stage_matches = []
        next_round = []

        for i in range(0, len(current_round), 2):
            if i + 1 >= len(current_round):
                next_round.append(current_round[i])
                continue
            home, away = current_round[i], current_round[i + 1]
            result = simulate_match(
                home, away, model_home, model_away, feature_cols,
                team_stats, results_df, is_neutral=True, allow_draw=False)
            stage_matches.append(result)
            next_round.append(result["winner"])

        bracket[stage] = stage_matches
        current_round = next_round

    if current_round:
        bracket["Champion"] = current_round[0]
        logger.info(f"\n CAMPEON PREDICHO: {current_round[0]}")

    return bracket


def monte_carlo_simulation(groups, model_home, model_away, feature_cols,
                            team_stats, results_df, n_simulations=1000):
    logger.info(f"\nMonte Carlo: {n_simulations} simulaciones...")
    win_counts   = {}
    final_counts = {}

    for sim in range(n_simulations):
        if sim % 100 == 0:
            logger.info(f"  Simulacion {sim}/{n_simulations}")
        np.random.seed(sim)
        gr = simulate_group_stage(
            groups, model_home, model_away, feature_cols, team_stats, results_df)
        qualified = get_qualified_teams(gr)
        bracket   = simulate_knockout_stage(
            qualified, model_home, model_away, feature_cols, team_stats, results_df)

        champion = bracket.get("Champion")
        if champion:
            win_counts[champion] = win_counts.get(champion, 0) + 1
        if "Final" in bracket and bracket["Final"]:
            for team in [bracket["Final"][0]["home"], bracket["Final"][0]["away"]]:
                final_counts[team] = final_counts.get(team, 0) + 1

    all_teams = set(list(win_counts.keys()) + list(final_counts.keys()))
    rows = [{"team": t,
             "p_win_tournament": win_counts.get(t, 0) / n_simulations,
             "p_reach_final":    final_counts.get(t, 0) / n_simulations,
             "sim_wins":         win_counts.get(t, 0),
             "sim_finals":       final_counts.get(t, 0)}
            for t in sorted(all_teams)]

    return (pd.DataFrame(rows)
              .sort_values("p_win_tournament", ascending=False)
              .reset_index(drop=True)
              .assign(rank=lambda df: range(1, len(df) + 1)))


def run_tournament_simulation(n_monte_carlo=0, groups=None):
    model_home, model_away, feature_cols = load_trained_models()
    results_df, rankings = load_raw_data()
    team_stats = build_team_stats(results_df, rankings)

    if groups is None:
        groups = get_groups()

    # Fase de grupos
    group_results = simulate_group_stage(
        groups, model_home, model_away, feature_cols, team_stats, results_df)

    all_tables  = []
    all_matches = []
    for gname, gdata in group_results.items():
        print(f"\nGrupo {gname}:")
        print(gdata["table"][["team", "pts", "gf", "gc", "gd"]].to_string(index=False))
        all_tables.append(gdata["table"])
        all_matches.append(gdata["matches"])

    save_results(pd.concat(all_tables,  ignore_index=True), "group_stage_standings.csv")
    save_results(pd.concat(all_matches, ignore_index=True), "group_stage_matches.csv")

    qualified = get_qualified_teams(group_results)
    bracket   = simulate_knockout_stage(
        qualified, model_home, model_away, feature_cols, team_stats, results_df)

    champion = bracket.get("Champion", "Unknown")
    save_results(f"Campeon predicho: {champion}\n", "tournament_winner.txt")
    print_bracket(bracket)

    bracket_rows = []
    for stage, matches in bracket.items():
        if stage == "Champion" or not isinstance(matches, list):
            continue
        for m in matches:
            bracket_rows.append({**m, "stage": stage})
    if bracket_rows:
        save_results(pd.DataFrame(bracket_rows), "knockout_bracket.csv")

    output = {"group_results": group_results, "qualified": qualified,
              "bracket": bracket, "champion": champion}

    if n_monte_carlo > 0:
        mc_df = monte_carlo_simulation(
            groups, model_home, model_away, feature_cols,
            team_stats, results_df, n_simulations=n_monte_carlo)
        save_results(mc_df, "monte_carlo_probabilities.csv")
        print("\nTOP 10 FAVORITOS (Monte Carlo):")
        print(mc_df.head(10)[["rank", "team", "p_win_tournament", "p_reach_final"]].to_string(index=False))
        output["monte_carlo"] = mc_df

    return output


if __name__ == "__main__":
    run_tournament_simulation()
