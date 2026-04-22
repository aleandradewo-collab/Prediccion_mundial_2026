"""
tournament.py - Simulacion completa del Mundial 2026.

Formato 2026:
  - 48 equipos, 12 grupos de 4
  - Fase de grupos: todos contra todos (3 partidos por equipo)
  - Clasifican: 2 primeros de cada grupo + 8 mejores terceros = 32 equipos
  - Ronda 1/32 -> 1/16 -> Cuartos -> Semis -> Final
  - Empates en eliminatorias -> tanda de penaltis con probabilidades reales
    basadas en el historial de shootouts.csv de cada equipo
"""

import logging
import numpy as np
import pandas as pd
from typing import Optional

from src.utils import (
    HOST_NATIONS, DATA_PROCESSED, DATA_RAW,
    logger, get_groups, print_bracket, save_results
)
from src.model import predict_goals, load_trained_models
from src.data_preparation import build_team_stats, load_raw_data, compute_h2h
from src.dixon_coles import run_ratings_pipeline, compute_expected_goals


# ── Probabilidades de penaltis ────────────────────────────────────────────────

DEFAULT_PENALTY_RATE  = 0.50   # para equipos sin historial
REGRESSION_WEIGHT     = 3      # tandas "fantasma" bayesianas hacia la media
PENALTY_DECAY_WEEKS   = 52 * 6 # vida media del peso: 10 años


def load_penalty_rates() -> dict:
    """
    Calcula la tasa de victoria en tandas de penaltis combinando:

    1. Decaimiento temporal exponencial: tandas recientes pesan más.
       Peso = exp(-log(2) * semanas_atras / PENALTY_DECAY_WEEKS)
       Con 4 años de vida media, una tanda de hace 4 años vale el 50%
       de una reciente. Una de hace 8 años vale solo el 25%.

    2. Suavizado bayesiano: añade REGRESSION_WEIGHT tandas fantasma
       con resultado 0.50 para evitar tasas extremas con pocos datos.

    Fórmula final:
        tasa = (sum(w_i * resultado_i) + 3*0.5) / (sum(w_i) + 3)
    """
    path = DATA_RAW / "shootouts.csv"
    df   = pd.read_csv(path, parse_dates=["date"])

    reference_date = df["date"].max()

    records = {}
    for _, row in df.iterrows():
        home, away, winner = row["home_team"], row["away_team"], row["winner"]
        weeks_ago = max((reference_date - row["date"]).days / 7, 0)
        weight    = np.exp(-np.log(2) * weeks_ago / PENALTY_DECAY_WEEKS)

        for team in [home, away]:
            if team not in records:
                records[team] = {"won_w": 0.0, "total_w": 0.0}
            records[team]["total_w"] += weight
            if team == winner:
                records[team]["won_w"] += weight

    rates = {}
    for team, r in records.items():
        smoothed = (r["won_w"] + REGRESSION_WEIGHT * DEFAULT_PENALTY_RATE) / \
                   (r["total_w"] + REGRESSION_WEIGHT)
        rates[team] = round(float(smoothed), 4)

    logger.info(f"Tasas de penaltis cargadas para {len(rates)} equipos "
                f"(decaimiento {PENALTY_DECAY_WEEKS} semanas)")
    return rates


def simulate_penalty_shootout(home: str, away: str, penalty_rates: dict) -> str:
    """
    Simula una tanda de penaltis usando las tasas reales de cada equipo.

    La probabilidad de que 'home' gane se calcula normalizando sus
    tasas relativas:
        p_home = rate_home / (rate_home + rate_away)

    Esto captura tanto "home es muy bueno" como "away es muy malo",
    de forma proporcional.

    Returns:
        Nombre del equipo ganador
    """
    rate_home = penalty_rates.get(home, DEFAULT_PENALTY_RATE)
    rate_away = penalty_rates.get(away, DEFAULT_PENALTY_RATE)

    # Normalizar para que sumen 1
    total = rate_home + rate_away
    if total == 0:
        p_home = 0.5
    else:
        p_home = rate_home / total

    return home if np.random.random() < p_home else away


# ── Features para predicción ──────────────────────────────────────────────────

def build_match_features_for_prediction(home, away, team_stats, results_df,
                                        is_neutral=False, ratings=None):
    def get_stats(team):
        if team in team_stats.index:
            s = team_stats.loc[team]
            return {
                "fifa_points":  float(s.get("fifa_points",    1200)),
                "fifa_rank":    float(s.get("fifa_rank",        50)),
                "avg_scored":   float(s.get("avg_goals_scored",  1.2)),
                "avg_conceded": float(s.get("avg_goals_conceded",1.2)),
                "win_rate":     float(s.get("win_rate",          0.33)),
            }
        return {"fifa_points": 1200, "fifa_rank": 50,
                "avg_scored": 1.2, "avg_conceded": 1.2, "win_rate": 0.33}

    h = get_stats(home)
    a = get_stats(away)
    compute_h2h(results_df, home, away)   # precalculo (no usado directamente aquí)

    def get_dc(team, col):
        if ratings is not None:
            row = ratings[ratings["team"] == team]
            if len(row) > 0:
                raw = float(row[col].values[0])
                # Mezcla con 1.0 (la media) para suavizar ratings extremos.
                # alpha=0.4 significa: 40% rating real, 60% media global.
                # Esto evita que defense=0.12 lleve los goles esperados a 0.
                alpha = 0.4
                return alpha * raw + (1 - alpha) * 1.0
        return 1.0

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
        "attack_rating_home":  get_dc(home, "attack_rating"),
        "attack_rating_away":  get_dc(away, "attack_rating"),
        "defense_rating_home": get_dc(home, "defense_rating"),
        "defense_rating_away": get_dc(away, "defense_rating"),
    }


# ── Simulación de partido ─────────────────────────────────────────────────────

def simulate_match(home, away, model_home, model_away, feature_cols,
                   team_stats, results_df, penalty_rates,
                   is_neutral=False, allow_draw=True, ratings=None):
    """
    Simula un partido usando distribución de Poisson para los goles.

    Si allow_draw=False y hay empate al final del tiempo reglamentario:
      → tanda de penaltis con probabilidades reales del historial de cada equipo

    Returns:
        dict con home, away, home_goals, away_goals, winner,
              went_to_penalties, penalty_winner (si aplica)
    """
    features = build_match_features_for_prediction(
        home, away, team_stats, results_df, is_neutral, ratings=ratings)
    lambda_home, lambda_away = predict_goals(
        model_home, model_away, feature_cols, features)

    hg = int(np.random.poisson(lambda_home))
    ag = int(np.random.poisson(lambda_away))

    went_to_penalties = False
    penalty_winner    = None

    if not allow_draw and hg == ag:
        went_to_penalties = True
        penalty_winner    = simulate_penalty_shootout(home, away, penalty_rates)
        winner            = penalty_winner
    elif hg > ag:
        winner = home
    elif ag > hg:
        winner = away
    else:
        winner = "Draw"

    return {
        "home":              home,
        "away":              away,
        "home_goals":        hg,
        "away_goals":        ag,
        "winner":            winner,
        "went_to_penalties": went_to_penalties,
        "penalty_winner":    penalty_winner,
        "lambda_home":       round(lambda_home, 3),
        "lambda_away":       round(lambda_away, 3),
    }


# ── Fase de grupos ────────────────────────────────────────────────────────────

def simulate_group_stage(groups, model_home, model_away, feature_cols,
                         team_stats, results_df, penalty_rates, ratings=None):
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
                    team_stats, results_df, penalty_rates,
                    is_neutral=True, allow_draw=True, ratings=ratings)

                hg, ag = result["home_goals"], result["away_goals"]
                standings[home]["gf"] += hg
                standings[home]["gc"] += ag
                standings[home]["gd"] += hg - ag
                standings[away]["gf"] += ag
                standings[away]["gc"] += hg
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
        table["group"]    = group_name

        group_results[group_name] = {
            "table":   table,
            "matches": pd.DataFrame(match_log),
        }

    return group_results


# ── Clasificados ──────────────────────────────────────────────────────────────

def get_classified(group_results):
    """
    Devuelve un dict con la clasificación completa:
      - firsts:  {grupo: equipo} — 1º de cada grupo
      - seconds: {grupo: equipo} — 2º de cada grupo
      - thirds:  lista de dicts con el 3º de cada grupo (ordenados por pts/gd/gf)

    Los 8 mejores terceros también clasifican.
    """
    firsts  = {}
    seconds = {}
    thirds  = []

    for gname, data in group_results.items():
        table = data["table"]
        firsts[gname]  = table.iloc[0]["team"]
        seconds[gname] = table.iloc[1]["team"]
        if len(table) > 2:
            row = table.iloc[2].to_dict()
            row["group"] = gname
            thirds.append(row)

    # Ordenar terceros y quedarse con los 8 mejores
    thirds_df = (pd.DataFrame(thirds)
                   .sort_values(["pts", "gd", "gf"], ascending=False)
                   .head(8)
                   .reset_index(drop=True))

    best_thirds = {}
    for _, row in thirds_df.iterrows():
        best_thirds[row["group"]] = row["team"]

    return firsts, seconds, best_thirds


def build_round_of_32(firsts, seconds, best_thirds):
    """
    Construye los 16 enfrentamientos de los 1/32 siguiendo las reglas:
      - Nunca 1º vs 1º
      - Nunca 3º vs 3º  -> los terceros siempre juegan contra primeros
      - 2º vs 2º sí está permitido
      - Nunca equipos del mismo grupo

    Con 12 primeros, 12 segundos y 8 terceros (32 equipos, 16 partidos):
      - 8 partidos de 1º vs 3º  (todos los terceros juegan contra primeros)
      - 4 partidos de 1º vs 2º  (primeros restantes vs segundos)
      - 4 partidos de 2º vs 2º  (segundos restantes entre sí)

    Asignación concreta (grupos A-L, terceros de los 8 mejores grupos):
      1ºI vs 3ºA,  1ºJ vs 3ºB,  1ºK vs 3ºC,  1ºL vs 3ºD   <- 1ºs sin tercero clasificado
      1ºA vs 3ºE,  1ºB vs 3ºF,  1ºC vs 3ºG,  1ºD vs 3ºH   <- 1ºs vs tercero de otro grupo
      1ºE vs 2ºH,  1ºF vs 2ºG,  1ºG vs 2ºF,  1ºH vs 2ºE   <- 1ºvs2º (grupos distintos)
      2ºA vs 2ºB,  2ºC vs 2ºD,  2ºI vs 2ºJ,  2ºK vs 2ºL   <- 2ºvs2º
    """
    all_groups   = ["A","B","C","D","E","F","G","H","I","J","K","L"]
    third_groups = sorted(best_thirds.keys())  # grupos cuyos terceros clasificaron

    # Grupos cuyos terceros NO clasificaron (jugarán vs terceros de otros grupos)
    no_third_groups = [g for g in all_groups if g not in third_groups]

    # ── 8 partidos 1º vs 3º ──────────────────────────────────────────────────
    # Los primeros de grupos sin tercero clasificado (I,J,K,L típicamente)
    # juegan contra los 4 mejores terceros (evitando mismo grupo)
    # Los primeros de A,B,C,D juegan contra los otros 4 terceros (E,F,G,H)
    matchups_1v3 = []
    thirds_sorted = sorted(best_thirds.keys())  # orden por rendimiento ya hecho en get_classified

    # Asignar terceros a primeros de grupos sin tercero propio primero
    assigned_thirds = set()
    for first_group in no_third_groups:
        for third_group in thirds_sorted:
            if third_group not in assigned_thirds and third_group != first_group:
                matchups_1v3.append((firsts[first_group], best_thirds[third_group]))
                assigned_thirds.add(third_group)
                break

    # Asignar terceros restantes a primeros de los otros grupos (evitando mismo grupo)
    remaining_thirds = [g for g in thirds_sorted if g not in assigned_thirds]
    used_firsts = set(no_third_groups)
    for third_group in remaining_thirds:
        for first_group in all_groups:
            if first_group not in used_firsts and first_group != third_group:
                matchups_1v3.append((firsts[first_group], best_thirds[third_group]))
                used_firsts.add(first_group)
                assigned_thirds.add(third_group)
                break

    # ── 4 partidos 1º vs 2º ──────────────────────────────────────────────────
    # Los primeros que NO jugaron vs terceros, vs segundos de grupos distintos
    firsts_for_1v2  = [g for g in all_groups if g not in used_firsts]
    seconds_for_1v2 = []
    used_seconds    = set()

    matchups_1v2 = []
    for fg in firsts_for_1v2:
        for sg in all_groups:
            if sg not in used_seconds and sg != fg:
                matchups_1v2.append((firsts[fg], seconds[sg]))
                used_seconds.add(sg)
                break

    # ── 4 partidos 2º vs 2º ──────────────────────────────────────────────────
    remaining_seconds = [g for g in all_groups if g not in used_seconds]
    matchups_2v2 = []
    paired = set()
    for i, g1 in enumerate(remaining_seconds):
        if g1 in paired:
            continue
        for g2 in remaining_seconds[i+1:]:
            if g2 not in paired and g2 != g1:
                matchups_2v2.append((seconds[g1], seconds[g2]))
                paired.add(g1)
                paired.add(g2)
                break

    matchups = matchups_1v3 + matchups_1v2 + matchups_2v2

    logger.info(f"  Bracket 1/32: {len(matchups)} partidos "
                f"(1ºvs3º: {len(matchups_1v3)}, "
                f"1ºvs2º: {len(matchups_1v2)}, "
                f"2ºvs2º: {len(matchups_2v2)})")
    return matchups


def get_qualified_teams(group_results):
    """Mantiene compatibilidad con el resto del código — devuelve lista de 32."""
    firsts, seconds, best_thirds = get_classified(group_results)
    matchups = build_round_of_32(firsts, seconds, best_thirds)

    # Aplanar en lista ordenada para el bracket
    ordered = []
    for home, away in matchups:
        ordered.append(home)
        ordered.append(away)

    logger.info(f"Equipos clasificados: {len(ordered)//2*2} en {len(matchups)} partidos")
    return ordered[:32]


# ── Fase eliminatoria ─────────────────────────────────────────────────────────

def simulate_knockout_stage(qualified, model_home, model_away, feature_cols,
                             team_stats, results_df, penalty_rates, ratings=None):
    logger.info("\nSimulando fase eliminatoria...")
    bracket      = {}
    current_round = qualified.copy()
    stage_names  = ["Round of 32", "Round of 16",
                    "Quarter-finals", "Semi-finals", "Final"]

    for stage in stage_names:
        if len(current_round) < 2:
            break
        logger.info(f"  -> {stage} ({len(current_round)} equipos)")
        stage_matches = []
        next_round    = []

        for i in range(0, len(current_round), 2):
            if i + 1 >= len(current_round):
                next_round.append(current_round[i])
                continue
            home, away = current_round[i], current_round[i + 1]
            result = simulate_match(
                home, away, model_home, model_away, feature_cols,
                team_stats, results_df, penalty_rates,
                is_neutral=True, allow_draw=False, ratings=ratings)

            # Log si fue a penaltis
            if result["went_to_penalties"]:
                logger.info(
                    f"     *** PENALTIS: {home} vs {away} "
                    f"-> gana {result['penalty_winner']}"
                )

            stage_matches.append(result)
            next_round.append(result["winner"])

        bracket[stage]  = stage_matches
        current_round   = next_round

    if current_round:
        bracket["Champion"] = current_round[0]
        logger.info(f"\n CAMPEON PREDICHO: {current_round[0]}")

    return bracket


# ── Monte Carlo ───────────────────────────────────────────────────────────────

def monte_carlo_simulation(groups, model_home, model_away, feature_cols,
                            team_stats, results_df, penalty_rates,
                            ratings=None, n_simulations=1000):
    logger.info(f"\nMonte Carlo: {n_simulations} simulaciones...")
    win_counts   = {}
    final_counts = {}
    penalty_counts = {}   # cuántas veces cada equipo llegó a penaltis y ganó

    root_logger    = logging.getLogger()
    original_level = root_logger.level
    root_logger.setLevel(logging.WARNING)

    for sim in range(n_simulations):
        if sim % 10 == 0:
            pct = int((sim / n_simulations) * 30)
            bar = "#" * pct + "." * (30 - pct)
            print(f"\r  [{bar}] {sim}/{n_simulations}", end="", flush=True)

        np.random.seed(sim)
        gr = simulate_group_stage(
            groups, model_home, model_away, feature_cols,
            team_stats, results_df, penalty_rates, ratings=ratings)
        qualified = get_qualified_teams(gr)
        bracket   = simulate_knockout_stage(
            qualified, model_home, model_away, feature_cols,
            team_stats, results_df, penalty_rates, ratings=ratings)

        champion = bracket.get("Champion")
        if champion:
            win_counts[champion] = win_counts.get(champion, 0) + 1

        if "Final" in bracket and bracket["Final"]:
            for team in [bracket["Final"][0]["home"], bracket["Final"][0]["away"]]:
                final_counts[team] = final_counts.get(team, 0) + 1

        # Contar victorias vía penaltis en toda la fase eliminatoria
        for stage, matches in bracket.items():
            if not isinstance(matches, list):
                continue
            for m in matches:
                if m.get("went_to_penalties") and m.get("penalty_winner"):
                    pw = m["penalty_winner"]
                    penalty_counts[pw] = penalty_counts.get(pw, 0) + 1

    root_logger.setLevel(original_level)
    bar = "#" * 30
    print(f"\r  [{bar}] {n_simulations}/{n_simulations} completadas.")

    all_teams = set(list(win_counts.keys()) + list(final_counts.keys()))
    rows = []
    for t in sorted(all_teams):
        rows.append({
            "team":               t,
            "p_win_tournament":   win_counts.get(t, 0)   / n_simulations,
            "p_reach_final":      final_counts.get(t, 0) / n_simulations,
            "penalty_wins":       penalty_counts.get(t, 0),
            "sim_wins":           win_counts.get(t, 0),
            "sim_finals":         final_counts.get(t, 0),
        })

    return (pd.DataFrame(rows)
              .sort_values("p_win_tournament", ascending=False)
              .reset_index(drop=True)
              .assign(rank=lambda df: range(1, len(df) + 1)))


# ── Pipeline principal ────────────────────────────────────────────────────────

def run_tournament_simulation(n_monte_carlo=1000, groups=None):
    model_home, model_away, feature_cols = load_trained_models()
    results_df, rankings = load_raw_data()
    team_stats = build_team_stats(results_df, rankings)

    # Cargar ratings Dixon-Coles (desde CSV si ya existen, si no recalcular)
    ratings_path = DATA_PROCESSED / "attack_defense_ratings.csv"
    if ratings_path.exists():
        ratings = pd.read_csv(ratings_path)
        logger.info(f"Ratings cargados desde {ratings_path}")
    else:
        ratings = run_ratings_pipeline(results_df)

    # Cargar tasas de penaltis reales
    penalty_rates = load_penalty_rates()

    # Mostrar tasas de equipos del Mundial para información
    logger.info("\nTasas de penaltis de los equipos del Mundial (suavizado bayesiano):")
    if groups is None:
        groups = get_groups()
    all_wc_teams = [t for grp in groups.values() for t in grp]
    for team in sorted(all_wc_teams):
        rate = penalty_rates.get(team, DEFAULT_PENALTY_RATE)
        logger.info(f"  {team:30s} {rate:.3f}")

    # Simulacion base (bracket único con semilla fija)
    group_results = simulate_group_stage(
        groups, model_home, model_away, feature_cols,
        team_stats, results_df, penalty_rates, ratings=ratings)

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
        qualified, model_home, model_away, feature_cols,
        team_stats, results_df, penalty_rates, ratings=ratings)

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

    output = {
        "group_results": group_results,
        "qualified":     qualified,
        "bracket":       bracket,
        "champion":      champion,
        "penalty_rates": penalty_rates,
    }

    # Monte Carlo
    if n_monte_carlo > 1:
        mc_df = monte_carlo_simulation(
            groups, model_home, model_away, feature_cols,
            team_stats, results_df, penalty_rates,
            ratings=ratings, n_simulations=n_monte_carlo)
        save_results(mc_df, "monte_carlo_probabilities.csv")
        output["monte_carlo"] = mc_df

    return output


if __name__ == "__main__":
    run_tournament_simulation()
