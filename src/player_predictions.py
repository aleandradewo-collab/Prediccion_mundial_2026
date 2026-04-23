"""
player_predictions.py - Predicciones individuales de jugadores.

En cada simulación del torneo rastrea:
  - Goles por jugador partido a partido
  - Asistencias por jugador
  - Calcula al final: máximo goleador, máximo asistente, mejor jugador (MVP)

El MVP se determina por una fórmula combinada:
  mvp_score = goles * 2.5 + asistencias * 1.5 + (rondas_jugadas * 0.5)

En Monte Carlo, tras N simulaciones devuelve:
  - P(jugador = máximo goleador)
  - Goles esperados en el torneo
  - P(jugador = MVP)
"""

import numpy as np
import pandas as pd
from src.utils import DATA_PROCESSED, logger


def get_team_scorers(team: str, player_df: pd.DataFrame,
                     top_n: int = 5) -> list[dict]:
    """
    Devuelve los top_n jugadores de un equipo que pueden marcar goles,
    con su probabilidad relativa de ser el goleador en un partido.

    La probabilidad se pondera por:
      - Valor de mercado (proxy de calidad)
      - form_score (rendimiento reciente)
      - Posición (delanteros tienen más peso)

    Returns:
        Lista de dicts con: name, p_score (prob de marcar un gol del equipo),
                            p_assist (prob de dar una asistencia)
    """
    tm = player_df[player_df["country_of_citizenship"] == team].copy()
    if len(tm) == 0:
        return []

    # Tomar los mejores jugadores por valor
    tm = tm.nlargest(min(23, len(tm)), "market_value_in_eur")

    # Peso base por posición para goles
    position_goal_weight = {
        "Attack":     3.0,
        "Midfield":   1.2,
        "Defender":   0.4,
        "Goalkeeper": 0.05,
    }
    position_assist_weight = {
        "Attack":     1.5,
        "Midfield":   2.0,
        "Defender":   0.8,
        "Goalkeeper": 0.1,
    }

    tm["pos_goal_w"]   = tm["position"].map(position_goal_weight).fillna(1.0)
    tm["pos_assist_w"] = tm["position"].map(position_assist_weight).fillna(1.0)

    # Score combinado para goles
    val_norm = tm["market_value_in_eur"] / max(tm["market_value_in_eur"].max(), 1)
    form_norm = tm["form_score"] / max(tm["form_score"].max(), 1)

    tm["goal_score"]   = (val_norm * 0.4 + form_norm * 0.6) * tm["pos_goal_w"]
    tm["assist_score"] = (val_norm * 0.4 + form_norm * 0.6) * tm["pos_assist_w"]

    # Normalizar a probabilidades
    total_g = tm["goal_score"].sum()
    total_a = tm["assist_score"].sum()

    if total_g == 0:
        tm["p_score"]  = 1.0 / len(tm)
        tm["p_assist"] = 1.0 / len(tm)
    else:
        tm["p_score"]  = tm["goal_score"]  / total_g
        tm["p_assist"] = tm["assist_score"] / total_a

    # Devolver top_n goleadores potenciales
    top = tm.nlargest(top_n, "goal_score")
    return [
        {
            "name":      row["name"],
            "team":      team,
            "p_score":   float(row["p_score"]),
            "p_assist":  float(row["p_assist"]),
            "position":  row.get("position", "Unknown"),
            "value_M":   row["market_value_in_eur"] / 1e6,
        }
        for _, row in top.iterrows()
    ]


def simulate_match_scorers(home: str, away: str,
                            home_goals: int, away_goals: int,
                            player_df: pd.DataFrame,
                            scorers_cache: dict) -> dict:
    """
    Dado un marcador ya simulado (home_goals, away_goals),
    distribuye los goles y asistencias entre los jugadores del equipo.

    Usa un cache de scorers por equipo para no recalcular en cada partido.

    Returns:
        dict con listas de goleadores y asistentes del partido
    """
    result = {"goals": [], "assists": []}

    for team, n_goals in [(home, home_goals), (away, away_goals)]:
        if team not in scorers_cache:
            scorers_cache[team] = get_team_scorers(team, player_df)

        scorers = scorers_cache[team]
        if not scorers or n_goals == 0:
            continue

        names      = [s["name"]     for s in scorers]
        p_scores   = [s["p_score"]  for s in scorers]
        p_assists  = [s["p_assist"] for s in scorers]

        # Normalizar por si no suman exactamente 1
        p_scores  = np.array(p_scores)  / sum(p_scores)
        p_assists = np.array(p_assists) / sum(p_assists)

        # Muestrear goleadores (con reemplazo — un jugador puede marcar 2)
        for _ in range(n_goals):
            scorer  = np.random.choice(names, p=p_scores)
            assister_pool = [n for n in names if n != scorer]
            if assister_pool and np.random.random() < 0.75:
                assist_p = p_assists[[i for i,n in enumerate(names) if n != scorer]]
                assist_p = assist_p / assist_p.sum()
                assister = np.random.choice(assister_pool, p=assist_p)
            else:
                assister = None

            result["goals"].append({"player": scorer,   "team": team})
            if assister:
                result["assists"].append({"player": assister, "team": team})

    return result


def aggregate_tournament_stats(match_stats: list) -> pd.DataFrame:
    """
    Agrega estadísticas de todos los partidos del torneo en una tabla
    de jugadores con goles, asistencias y mvp_score.

    Args:
        match_stats: lista de dicts devueltos por simulate_match_scorers

    Returns:
        DataFrame con columnas: player, team, goals, assists, mvp_score
    """
    goals_count   = {}
    assists_count = {}
    team_map      = {}

    for match in match_stats:
        for g in match.get("goals", []):
            key = g["player"]
            goals_count[key]   = goals_count.get(key, 0) + 1
            team_map[key]      = g["team"]
        for a in match.get("assists", []):
            key = a["player"]
            assists_count[key] = assists_count.get(key, 0) + 1
            team_map[key]      = a["team"]

    all_players = set(goals_count) | set(assists_count)
    rows = []
    for p in all_players:
        g = goals_count.get(p, 0)
        a = assists_count.get(p, 0)
        rows.append({
            "player":    p,
            "team":      team_map.get(p, ""),
            "goals":     g,
            "assists":   a,
            "mvp_score": g * 2.5 + a * 1.5,
        })

    return pd.DataFrame(rows).sort_values("goals", ascending=False).reset_index(drop=True)


def print_tournament_awards(stats: pd.DataFrame, champion: str):
    """Imprime los premios individuales del torneo."""
    print("\n" + "="*55)
    print("  PREMIOS INDIVIDUALES DEL TORNEO")
    print("="*55)

    top_scorer = stats.nlargest(1, "goals").iloc[0]
    top_assist = stats.nlargest(1, "assists").iloc[0]
    mvp        = stats.nlargest(1, "mvp_score").iloc[0]

    print(f"  🥇 Campeón:          {champion}")
    print(f"  ⚽ Máximo goleador:  {top_scorer['player']} ({top_scorer['team']}) "
          f"— {top_scorer['goals']} goles")
    print(f"  🎯 Máx. asistente:  {top_assist['player']} ({top_assist['team']}) "
          f"— {top_assist['assists']} asistencias")
    print(f"  🏆 MVP (Balón Oro):  {mvp['player']} ({mvp['team']}) "
          f"— score={mvp['mvp_score']:.1f}")
    print("="*55)


def monte_carlo_player_stats(all_sim_stats: list, n_simulations: int) -> pd.DataFrame:
    """
    Agrega estadísticas de jugadores a través de N simulaciones Monte Carlo.

    Args:
        all_sim_stats: lista de DataFrames (uno por simulación)
        n_simulations: número total de simulaciones

    Returns:
        DataFrame con probabilidades por jugador
    """
    goals_total     = {}
    assists_total   = {}
    top_scorer_wins = {}
    mvp_wins        = {}
    team_map        = {}

    for sim_df in all_sim_stats:
        if sim_df is None or len(sim_df) == 0:
            continue

        for _, row in sim_df.iterrows():
            p = row["player"]
            goals_total[p]   = goals_total.get(p, 0)   + row["goals"]
            assists_total[p] = assists_total.get(p, 0) + row["assists"]
            team_map[p]      = row["team"]

        # Top goleador de esta simulación
        top = sim_df.nlargest(1, "goals")
        if len(top) > 0:
            ts = top.iloc[0]["player"]
            top_scorer_wins[ts] = top_scorer_wins.get(ts, 0) + 1

        # MVP de esta simulación
        mvp_sim = sim_df.nlargest(1, "mvp_score")
        if len(mvp_sim) > 0:
            mv = mvp_sim.iloc[0]["player"]
            mvp_wins[mv] = mvp_wins.get(mv, 0) + 1

    all_players = set(goals_total) | set(assists_total)
    rows = []
    for p in all_players:
        rows.append({
            "player":             p,
            "team":               team_map.get(p, ""),
            "avg_goals":          round(goals_total.get(p, 0)   / n_simulations, 3),
            "avg_assists":        round(assists_total.get(p, 0) / n_simulations, 3),
            "p_top_scorer":       round(top_scorer_wins.get(p, 0) / n_simulations, 4),
            "p_mvp":              round(mvp_wins.get(p, 0)        / n_simulations, 4),
            "sim_top_scorer":     top_scorer_wins.get(p, 0),
            "sim_mvp":            mvp_wins.get(p, 0),
        })

    df = pd.DataFrame(rows).sort_values("avg_goals", ascending=False).reset_index(drop=True)
    df["rank_goals"] = range(1, len(df) + 1)
    return df


if __name__ == "__main__":
    from src.player_data import build_player_dataset
    player_df = build_player_dataset()
    scorers = get_team_scorers("France", player_df)
    print("Top goleadores Francia:")
    for s in scorers:
        print(f"  {s['name']:<25} p_gol={s['p_score']:.3f}  valor={s['value_M']:.0f}M€")
