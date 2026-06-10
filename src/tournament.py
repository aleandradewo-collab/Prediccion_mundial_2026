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
from src.squad_strength import build_squad_features, get_squad_features_for_team
from src.player_data import build_player_dataset
from src.player_predictions import (
    simulate_match_scorers, aggregate_tournament_stats,
    print_tournament_awards, monte_carlo_player_stats
)


# ── Probabilidades de penaltis ────────────────────────────────────────────────

DEFAULT_PENALTY_RATE  = 0.50   # para equipos sin historial
REGRESSION_WEIGHT     = 3      # tandas "fantasma" bayesianas hacia la media
PENALTY_DECAY_WEEKS   = 52 * 4 # vida media del peso: 4 años


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
                                        is_neutral=False, ratings=None, squad_df=None):
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
                # alpha=0.75: 75% rating real, 25% media global.
                # Subido desde 0.4 para que los ratings Dixon-Coles diferencien
                # mejor entre equipos fuertes y débiles en la simulación.
                # Con alpha=0.4 España (2.24) quedaba en 1.50 y Haití (0.5) en 0.80
                # — diferencia aplastada. Con 0.75: España=1.93, Haití=0.625.
                alpha = 0.75
                return alpha * raw + (1 - alpha) * 1.0
        return 1.0

    # Squad features (Transfermarkt)
    def get_sq(team, key):
        if squad_df is not None:
            sf = get_squad_features_for_team(team, squad_df)
            return sf.get(key, 0.0)
        return 0.0

    sv_home = get_sq(home, "squad_value_M")
    sv_away = get_sq(away, "squad_value_M")

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
        "squad_value_home":    sv_home,
        "squad_value_away":    sv_away,
        "squad_form_home":     get_sq(home, "squad_form_score"),
        "squad_form_away":     get_sq(away, "squad_form_score"),
        "squad_age_home":      get_sq(home, "squad_avg_age"),
        "squad_age_away":      get_sq(away, "squad_avg_age"),
        "top_scorer_val_home": get_sq(home, "top_scorer_value_M"),
        "top_scorer_val_away": get_sq(away, "top_scorer_value_M"),
        "wc_goals_home":       get_sq(home, "wc_goals_weighted"),
        "wc_goals_away":       get_sq(away, "wc_goals_weighted"),
        "value_ratio":         (sv_home + 1) / (sv_away + 1),
    }


# ── Simulación de partido ─────────────────────────────────────────────────────

def simulate_match(home, away, model_home, model_away, feature_cols,
                   team_stats, results_df, penalty_rates,
                   is_neutral=False, allow_draw=True, ratings=None, squad_df=None):
    """
    Simula un partido usando distribución de Poisson para los goles.

    Si allow_draw=False y hay empate al final del tiempo reglamentario:
      → tanda de penaltis con probabilidades reales del historial de cada equipo

    Returns:
        dict con home, away, home_goals, away_goals, winner,
              went_to_penalties, penalty_winner (si aplica)
    """
    features = build_match_features_for_prediction(
        home, away, team_stats, results_df, is_neutral, ratings=ratings, squad_df=squad_df)
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
                         team_stats, results_df, penalty_rates, ratings=None, squad_df=None, player_df=None):
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
                    is_neutral=True, allow_draw=True, ratings=ratings, squad_df=squad_df)

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

                # Rastrear goles individuales si hay datos de jugadores
                if player_df is not None:
                    scorer_data = simulate_match_scorers(
                        home, away, result["home_goals"], result["away_goals"],
                        player_df, scorers_cache if "scorers_cache" in dir() else {})
                    result["scorer_data"] = scorer_data
                match_log.append({"group": group_name, **{k:v for k,v in result.items() if k != "scorer_data"}})
                if "all_match_scorers" not in dir():
                    pass
                else:
                    all_match_scorers.append(result.get("scorer_data", {}))

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



# ── Bracket oficial FIFA World Cup 2026 ──────────────────────────────────────
#
# Los 16 cruces de 1/32 están FIJOS y predeterminados. El camino a la final
# está sellado en 4 ramas. Estructura de la imagen oficial:
#
# RAMA 1 → QF1 → SF1        RAMA 3 → QF3 → SF2
#   M1:  1E  vs 3[ABCDF]      M9:  1C  vs 2F
#   M2:  1I  vs 3[CDFGH]      M10: 2E  vs 2I
#   M3:  2A  vs 2B             M11: 1A  vs 3[CEFHI]
#   M4:  1F  vs 2C             M12: 1L  vs 3[EHIJK]
#
# RAMA 2 → QF2 → SF1        RAMA 4 → QF4 → SF2
#   M5:  2K  vs 2L             M13: 1J  vs 2H
#   M6:  1H  vs 2J             M14: 2D  vs 2G
#   M7:  1D  vs 3[BEFIJ]      M15: 1B  vs 3[EFGIJ]
#   M8:  1G  vs 3[AEHIJ]      M16: 1K  vs 3[DEIJL]
#
# SF1 = winner(QF1) vs winner(QF2)
# SF2 = winner(QF3) vs winner(QF4)
# FINAL = winner(SF1) vs winner(SF2)

THIRD_SLOTS = {
    "ABCDF": ["A","B","C","D","F"],
    "CDFGH": ["C","D","F","G","H"],
    "BEFIJ": ["B","E","F","I","J"],
    "AEHIJ": ["A","E","H","I","J"],
    "CEFHI": ["C","E","F","H","I"],
    "EHIJK": ["E","H","I","J","K"],
    "EFGIJ": ["E","F","G","I","J"],
    "DEIJL": ["D","E","I","J","L"],
}


def assign_thirds_to_slots(best_thirds: dict) -> dict:
    """
    Asigna cada tercero clasificado a su slot del bracket oficial.
    Cada slot acepta terceros de ciertos grupos (THIRD_SLOTS).
    Se asigna el mejor tercero disponible a cada slot en orden de ranking.
    """
    thirds_ranked = list(best_thirds.items())
    slot_assignments = {}
    assigned_groups  = set()

    for slot_name, valid_groups in THIRD_SLOTS.items():
        for group, team in thirds_ranked:
            if group not in assigned_groups and group in valid_groups:
                slot_assignments[slot_name] = team
                assigned_groups.add(group)
                break
        if slot_name not in slot_assignments:
            # Fallback: cualquier tercero no asignado aún
            for group, team in thirds_ranked:
                if group not in assigned_groups:
                    slot_assignments[slot_name] = team
                    assigned_groups.add(group)
                    break

    return slot_assignments


def build_official_bracket(firsts: dict, seconds: dict, best_thirds: dict) -> list:
    """
    Construye los 16 enfrentamientos de 1/32 siguiendo el bracket oficial
    FIFA World Cup 2026 con posiciones FIJAS.

    Los partidos están agrupados de 4 en 4 por rama:
      [M1,M2,M3,M4]   = RAMA 1 (→ QF1 → SF1)
      [M5,M6,M7,M8]   = RAMA 2 (→ QF2 → SF1)
      [M9,M10,M11,M12]= RAMA 3 (→ QF3 → SF2)
      [M13,M14,M15,M16]= RAMA 4 (→ QF4 → SF2)

    El camino a la final está sellado: los equipos de SF1 (ramas 1+2)
    solo pueden cruzarse con equipos de SF2 (ramas 3+4) en la FINAL.
    """
    t = assign_thirds_to_slots(best_thirds)

    def f(g):    return firsts.get(g,  f"1_{g}")
    def s(g):    return seconds.get(g, f"2_{g}")
    def th(slot):return t.get(slot,    f"3_{slot}")

    # RAMA 1
    M1  = (f("E"),  th("ABCDF"))
    M2  = (f("I"),  th("CDFGH"))
    M3  = (s("A"),  s("B"))
    M4  = (f("F"),  s("C"))
    # RAMA 2
    M5  = (s("K"),  s("L"))
    M6  = (f("H"),  s("J"))
    M7  = (f("D"),  th("BEFIJ"))
    M8  = (f("G"),  th("AEHIJ"))
    # RAMA 3
    M9  = (f("C"),  s("F"))
    M10 = (s("E"),  s("I"))
    M11 = (f("A"),  th("CEFHI"))
    M12 = (f("L"),  th("EHIJK"))
    # RAMA 4
    M13 = (f("J"),  s("H"))
    M14 = (s("D"),  s("G"))
    M15 = (f("B"),  th("EFGIJ"))
    M16 = (f("K"),  th("DEIJL"))

    matchups = [M1,M2,M3,M4, M5,M6,M7,M8, M9,M10,M11,M12, M13,M14,M15,M16]

    logger.info(f"  Bracket oficial 1/32: {len(matchups)} partidos en 4 ramas")
    for i, (rama, ms) in enumerate([(1,[M1,M2,M3,M4]),(2,[M5,M6,M7,M8]),
                                     (3,[M9,M10,M11,M12]),(4,[M13,M14,M15,M16])], 1):
        logger.info(f"  Rama {i}: " + " | ".join(f"{h} vs {a}" for h,a in ms))

    return matchups

def build_round_of_32(firsts, seconds, best_thirds):
    """
    DEPRECATED — mantenida por compatibilidad. Usar build_official_bracket().
    Redirige al bracket oficial FIFA 2026.
    """
    return build_official_bracket(firsts, seconds, best_thirds)

def get_qualified_teams(group_results):
    """
    Devuelve lista de 32 equipos ordenados según el bracket oficial FIFA 2026.
    El orden preserva la estructura de ramas: de 2 en 2 son los cruces de 1/32,
    de 4 en 4 son las ramas que comparten cuarto de final.
    """
    firsts, seconds, best_thirds = get_classified(group_results)
    matchups = build_official_bracket(firsts, seconds, best_thirds)

    ordered = []
    for home, away in matchups:
        ordered.append(home)
        ordered.append(away)

    logger.info(f"  32 equipos colocados en bracket oficial ({len(matchups)} partidos)")
    return ordered[:32]


# ── Fase eliminatoria ─────────────────────────────────────────────────────────

def simulate_knockout_stage(qualified, model_home, model_away, feature_cols,
                             team_stats, results_df, penalty_rates, ratings=None, squad_df=None, player_df=None):
    """
    Simula la fase eliminatoria respetando el bracket oficial FIFA 2026.

    El bracket tiene 4 ramas fijas. Los 32 equipos en 'qualified' están
    ordenados según el bracket oficial (de 2 en 2 = cruce de 1/32,
    de 4 en 4 = misma rama → mismo cuarto de final).

    Estructura de ramas selladas:
      Rama 1 (pos 0-7)  ┐
                         ├→ SF1 → Final
      Rama 2 (pos 8-15) ┘
      Rama 3 (pos 16-23)┐
                         ├→ SF2 → Final
      Rama 4 (pos 24-31)┘
    """
    logger.info("\nSimulando fase eliminatoria (bracket oficial FIFA 2026)...")
    bracket = {}

    def play_match(home, away):
        result = simulate_match(
            home, away, model_home, model_away, feature_cols,
            team_stats, results_df, penalty_rates,
            is_neutral=True, allow_draw=False, ratings=ratings, squad_df=squad_df)
        if result["went_to_penalties"]:
            logger.info(f"     *** PENALTIS: {home} vs {away} → {result['penalty_winner']}")
        return result

    # ── Round of 32 (16 partidos, 4 ramas de 4) ───────────────────────────────
    logger.info(f"  -> Round of 32 (32 equipos)")
    r32_matches = []
    r32_winners = []
    for i in range(0, 32, 2):
        home, away = qualified[i], qualified[i+1]
        res = play_match(home, away)
        r32_matches.append(res)
        r32_winners.append(res["winner"])
    bracket["Round of 32"] = r32_matches
    # r32_winners tiene 16 ganadores en orden: [w1,w2,w3,w4, w5,w6,w7,w8, w9,w10,w11,w12, w13,w14,w15,w16]
    # Cada grupo de 4 es una rama → los pares dentro de la rama se cruzan en QF

    # ── Round of 16 (8 partidos) ──────────────────────────────────────────────
    # Dentro de cada rama: w1 vs w2 y w3 vs w4 → ganadores van al mismo QF
    logger.info(f"  -> Round of 16 (16 equipos)")
    r16_matches = []
    r16_winners = []
    for i in range(0, 16, 2):
        home, away = r32_winners[i], r32_winners[i+1]
        res = play_match(home, away)
        r16_matches.append(res)
        r16_winners.append(res["winner"])
    bracket["Round of 16"] = r16_matches
    # r16_winners: [qf1a, qf1b, qf2a, qf2b, qf3a, qf3b, qf4a, qf4b]

    # ── Quarter-finals (4 partidos) ───────────────────────────────────────────
    # QF1 = r16_winners[0] vs r16_winners[1]  (rama 1)
    # QF2 = r16_winners[2] vs r16_winners[3]  (rama 2)
    # QF3 = r16_winners[4] vs r16_winners[5]  (rama 3)
    # QF4 = r16_winners[6] vs r16_winners[7]  (rama 4)
    logger.info(f"  -> Quarter-finals (8 equipos)")
    qf_matches = []
    qf_winners = []
    for i in range(0, 8, 2):
        home, away = r16_winners[i], r16_winners[i+1]
        res = play_match(home, away)
        qf_matches.append(res)
        qf_winners.append(res["winner"])
    bracket["Quarter-finals"] = qf_matches
    # qf_winners: [sf1a, sf1b, sf2a, sf2b]

    # ── Semi-finals (2 partidos) ──────────────────────────────────────────────
    # SF1 = winner(QF1, rama1) vs winner(QF2, rama2)
    # SF2 = winner(QF3, rama3) vs winner(QF4, rama4)
    logger.info(f"  -> Semi-finals (4 equipos)")
    sf_matches = []
    sf_winners = []
    for i in range(0, 4, 2):
        home, away = qf_winners[i], qf_winners[i+1]
        res = play_match(home, away)
        sf_matches.append(res)
        sf_winners.append(res["winner"])
    bracket["Semi-finals"] = sf_matches

    # ── Final ─────────────────────────────────────────────────────────────────
    logger.info(f"  -> Final (2 equipos): {sf_winners[0]} vs {sf_winners[1]}")
    final_res = play_match(sf_winners[0], sf_winners[1])
    bracket["Final"]    = [final_res]
    bracket["Champion"] = final_res["winner"]
    logger.info(f"\n  🏆 CAMPEÓN PREDICHO: {final_res['winner']}")

    return bracket


# ── Monte Carlo ───────────────────────────────────────────────────────────────

def monte_carlo_simulation(groups, model_home, model_away, feature_cols,
                            team_stats, results_df, penalty_rates,
                            ratings=None, squad_df=None, player_df=None,
                            n_simulations=1000):
    logger.info(f"\nMonte Carlo: {n_simulations} simulaciones...")
    win_counts     = {}
    final_counts   = {}
    penalty_counts = {}
    all_player_sim_stats = []   # estadísticas de jugadores por simulación
    scorers_cache  = {}         # cache compartido para no recalcular

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
            team_stats, results_df, penalty_rates,
            ratings=ratings, squad_df=squad_df, player_df=player_df)
        qualified = get_qualified_teams(gr)
        bracket   = simulate_knockout_stage(
            qualified, model_home, model_away, feature_cols,
            team_stats, results_df, penalty_rates,
            ratings=ratings, squad_df=squad_df, player_df=player_df)

        champion = bracket.get("Champion")
        if champion:
            win_counts[champion] = win_counts.get(champion, 0) + 1

        if "Final" in bracket and bracket["Final"]:
            for team in [bracket["Final"][0]["home"], bracket["Final"][0]["away"]]:
                final_counts[team] = final_counts.get(team, 0) + 1

        for stage, matches in bracket.items():
            if not isinstance(matches, list):
                continue
            for m in matches:
                if m.get("went_to_penalties") and m.get("penalty_winner"):
                    pw = m["penalty_winner"]
                    penalty_counts[pw] = penalty_counts.get(pw, 0) + 1

        # Estadísticas individuales de esta simulación
        if player_df is not None:
            sim_match_scorers = []
            all_matches_sim = []
            for gdata in gr.values():
                all_matches_sim.extend(gdata["matches"].to_dict("records"))
            for stage, matches in bracket.items():
                if isinstance(matches, list):
                    all_matches_sim.extend(matches)

            for m in all_matches_sim:
                sd = simulate_match_scorers(
                    m.get("home",""), m.get("away",""),
                    int(m.get("home_goals",0)), int(m.get("away_goals",0)),
                    player_df, scorers_cache)
                sim_match_scorers.append(sd)

            sim_stats = aggregate_tournament_stats(sim_match_scorers)
            all_player_sim_stats.append(sim_stats)

    root_logger.setLevel(original_level)
    bar = "#" * 30
    print(f"\r  [{bar}] {n_simulations}/{n_simulations} completadas.")

    # Tabla de equipos
    all_teams = set(list(win_counts.keys()) + list(final_counts.keys()))
    rows = []
    for t in sorted(all_teams):
        rows.append({
            "team":             t,
            "p_win_tournament": win_counts.get(t, 0)   / n_simulations,
            "p_reach_final":    final_counts.get(t, 0) / n_simulations,
            "penalty_wins":     penalty_counts.get(t, 0),
            "sim_wins":         win_counts.get(t, 0),
            "sim_finals":       final_counts.get(t, 0),
        })

    teams_df = (pd.DataFrame(rows)
                  .sort_values("p_win_tournament", ascending=False)
                  .reset_index(drop=True)
                  .assign(rank=lambda df: range(1, len(df) + 1)))

    # Tabla de jugadores
    player_mc_df = None
    if all_player_sim_stats:
        player_mc_df = monte_carlo_player_stats(all_player_sim_stats, n_simulations)

    return teams_df, player_mc_df


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

    # Cargar features de plantilla Transfermarkt
    squad_path = DATA_PROCESSED / "squad_features.csv"
    if squad_path.exists():
        squad_df = pd.read_csv(squad_path).set_index("team")
        logger.info(f"Squad features cargadas desde {squad_path}")
    else:
        squad_df = build_squad_features()

    # Cargar dataset de jugadores FILTRADO a convocados WC2026
    from src.squad_filter import build_wc2026_player_dataset
    wc_path = DATA_PROCESSED / "player_dataset_wc2026.csv"
    if wc_path.exists():
        player_df = pd.read_csv(wc_path)
        logger.info(f"Dataset WC2026 cargado: {len(player_df):,} jugadores convocados")
    else:
        logger.info("Generando dataset WC2026 por primera vez...")
        player_df = build_wc2026_player_dataset()

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
        team_stats, results_df, penalty_rates, ratings=ratings,
        squad_df=squad_df, player_df=player_df)

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
        team_stats, results_df, penalty_rates, ratings=ratings,
        squad_df=squad_df, player_df=player_df)

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

    # Tracking de goles individuales en la simulación base
    if player_df is not None:
        all_base_matches = []
        for gdata in group_results.values():
            all_base_matches.extend(gdata["matches"].to_dict("records"))
        for stage, matches in bracket.items():
            if isinstance(matches, list):
                all_base_matches.extend(matches)

        scorers_cache = {}
        base_match_scorers = []
        for m in all_base_matches:
            sd = simulate_match_scorers(
                m.get("home",""), m.get("away",""),
                int(m.get("home_goals",0)), int(m.get("away_goals",0)),
                player_df, scorers_cache)
            base_match_scorers.append(sd)

        base_player_stats = aggregate_tournament_stats(base_match_scorers)
        save_results(base_player_stats, "player_stats_base.csv")
        print_tournament_awards(base_player_stats, champion)
        output["player_stats"] = base_player_stats

    # Monte Carlo
    if n_monte_carlo > 1:
        mc_teams_df, mc_player_df = monte_carlo_simulation(
            groups, model_home, model_away, feature_cols,
            team_stats, results_df, penalty_rates,
            ratings=ratings, squad_df=squad_df, player_df=player_df,
            n_simulations=n_monte_carlo)
        save_results(mc_teams_df, "monte_carlo_probabilities.csv")
        output["monte_carlo"] = mc_teams_df

        if mc_player_df is not None:
            save_results(mc_player_df, "monte_carlo_player_stats.csv")
            output["monte_carlo_players"] = mc_player_df

            print("\n📊 TOP 10 MÁXIMOS GOLEADORES PROBABLES (Monte Carlo):")
            print(f"  {'#':<4} {'Jugador':<25} {'Equipo':<20} {'Goles/sim':>10}  {'P(top scorer)':>14}")
            print("  " + "-" * 78)
            for _, row in mc_player_df.head(10).iterrows():
                print(f"  {int(row['rank_goals']):<4} {row['player']:<25} {row['team']:<20} "
                      f"{row['avg_goals']:>10.2f}  {row['p_top_scorer']*100:>13.1f}%")

    return output


if __name__ == "__main__":
    run_tournament_simulation()
