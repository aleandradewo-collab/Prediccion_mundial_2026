"""
player_data.py - Carga y limpieza de datos de jugadores (Transfermarkt).

Responsabilidades:
  1. Cargar jugadores activos (last_season >= 2024)
  2. Normalizar nombres de países al formato del proyecto
  3. Calcular valor de mercado actual por jugador
  4. Calcular forma reciente: goles+asist ponderados por liga y recencia
  5. Extraer rendimiento en Mundiales 2018 y 2022 desde goalscorers.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
from src.utils import DATA_RAW, DATA_PROCESSED, IMPORTANT_TOURNAMENTS, logger

# ── Paths de los datos de Transfermarkt ──────────────────────────────────────
TM_DIR = DATA_RAW

# ── Año mínimo para considerar un jugador activo ─────────────────────────────
MIN_LAST_SEASON = 2025

# ── Vida media del peso temporal para forma reciente (en semanas) ─────────────
FORM_DECAY_WEEKS = 52 * 2   # 2 años: partidos recientes pesan mucho más

# ── Pesos por competición (competition_id de Transfermarkt) ──────────────────
COMPETITION_WEIGHTS = {
    "CL":   3.0,   # UEFA Champions League
    "GB1":  2.5,   # Premier League
    "ES1":  2.5,   # La Liga
    "L1":   2.5,   # Bundesliga
    "IT1":  2.5,   # Serie A
    "FR1":  2.5,   # Ligue 1
    "EL":   2.0,   # Europa League
    "PO1":  1.8,   # Liga Portugal
    "NL1":  1.8,   # Eredivisie
    "BE1":  1.5,   # Pro League Bélgica
    "TR1":  1.5,   # Süper Lig
    "SC1":  1.5,   # Scottish Premiership
    "SA1":  1.2,   # Saudi Pro League
    "MLS1": 1.2,   # MLS
    "BRA1": 1.5,   # Brasileirao
    "ARG1": 1.5,   # Liga Profesional Argentina
    "MEX1": 1.2,   # Liga MX
    "JAP1": 1.2,   # J1 League
    "RSK1": 1.2,   # K League
    "FIWC": 4.0,   # FIFA World Cup (máxima importancia)
    "EURO": 3.0,   # UEFA Euro
    "COPA": 2.5,   # Copa América
    "AFCN": 2.0,   # Africa Cup of Nations
}
DEFAULT_COMPETITION_WEIGHT = 1.0

# ── Mapeo de nombres de país (Transfermarkt → formato results.csv) ─────────────
COUNTRY_NAME_MAP = {
    "Bosnia-Herzegovina":    "Bosnia and Herzegovina",
    "Curacao":               "Curaçao",
    "Cote d'Ivoire":         "Ivory Coast",
    "Korea, South":          "South Korea",
    "Türkiye":               "Turkey",
    "United States":         "United States",
    "DR Congo":              "DR Congo",
    "Cape Verde Islands":    "Cape Verde",
}


def normalize_country(name: str) -> str:
    return COUNTRY_NAME_MAP.get(name, name)


def load_active_players() -> pd.DataFrame:
    """
    Carga jugadores activos (last_season >= MIN_LAST_SEASON).
    Normaliza nombres de país y devuelve columnas esenciales.
    """
    logger.info(f"Cargando jugadores activos (last_season >= {MIN_LAST_SEASON})...")
    players = pd.read_csv(TM_DIR / "players.csv", low_memory=False)

    active = players[players["last_season"] >= MIN_LAST_SEASON].copy()
    active["country_of_citizenship"] = active["country_of_citizenship"].map(
        lambda x: normalize_country(str(x)) if pd.notna(x) else x
    )
    active["market_value_in_eur"] = pd.to_numeric(
        active["market_value_in_eur"], errors="coerce"
    ).fillna(0)

    logger.info(f"  Jugadores activos: {len(active):,}")
    return active[[
        "player_id", "name", "country_of_citizenship",
        "position", "date_of_birth",
        "market_value_in_eur", "current_club_name",
        "current_national_team_id",
    ]]


def load_player_form(active_player_ids: list) -> pd.DataFrame:
    """
    Calcula la forma reciente de cada jugador a partir de appearances.csv.

    Para cada jugador suma goles y asistencias ponderados por:
      - Importancia de la competición (COMPETITION_WEIGHTS)
      - Recencia temporal (decaimiento exponencial FORM_DECAY_WEEKS)

    Solo considera apariciones desde 2022 para mantener relevancia.

    Returns:
        DataFrame con columnas: player_id, weighted_goals, weighted_assists,
                                weighted_minutes, form_score
    """
    logger.info("Calculando forma reciente de jugadores...")

    # Leer solo columnas necesarias para ahorrar memoria
    app = pd.read_csv(
        TM_DIR / "appearances.csv",
        usecols=["player_id", "date", "competition_id",
                 "goals", "assists", "minutes_played"],
        parse_dates=["date"],
        low_memory=False,
    )

    # Filtrar solo jugadores activos y desde 2022
    app = app[
        (app["player_id"].isin(active_player_ids)) &
        (app["date"] >= "2022-01-01")
    ].copy()

    reference_date = app["date"].max()

    # Peso temporal
    app["weeks_ago"] = (reference_date - app["date"]).dt.days / 7
    app["time_weight"] = np.exp(
        -np.log(2) * app["weeks_ago"] / FORM_DECAY_WEEKS
    )

    # Peso por competición
    app["comp_weight"] = app["competition_id"].map(
        COMPETITION_WEIGHTS
    ).fillna(DEFAULT_COMPETITION_WEIGHT)

    app["weight"] = app["time_weight"] * app["comp_weight"]

    # Agregar por jugador
    form = app.groupby("player_id").apply(
        lambda g: pd.Series({
            "weighted_goals":   (g["goals"]   * g["weight"]).sum(),
            "weighted_assists": (g["assists"] * g["weight"]).sum(),
            "weighted_minutes": (g["minutes_played"] * g["weight"]).sum(),
            "total_apps":       len(g),
        })
    ).reset_index()

    # form_score: métrica combinada de rendimiento
    form["form_score"] = (
        form["weighted_goals"]   * 2.0 +
        form["weighted_assists"] * 1.5 +
        form["weighted_minutes"] / 90 * 0.1
    )

    logger.info(f"  Forma calculada para {len(form):,} jugadores")
    return form


def load_world_cup_history() -> pd.DataFrame:
    """
    Extrae el rendimiento de cada jugador en los Mundiales 2018 y 2022
    a partir de goalscorers.csv del proyecto.

    Peso temporal: 2022 vale el doble que 2018.

    Returns:
        DataFrame con columnas: player_name, team, wc_goals_weighted, wc_appearances
    """
    logger.info("Cargando historial de Mundiales 2018 y 2022...")

    gs = pd.read_csv(DATA_RAW / "goalscorers.csv", parse_dates=["date"])
    results = pd.read_csv(DATA_RAW / "results.csv", parse_dates=["date"])

    # Fechas de partidos de Mundial 2018 y 2022
    wc = results[results["tournament"] == "FIFA World Cup"].copy()
    wc_2018 = set(wc[wc["date"].dt.year == 2018]["date"].astype(str))
    wc_2022 = set(wc[wc["date"].dt.year == 2022]["date"].astype(str))

    gs["date_str"] = gs["date"].astype(str)
    gs_wc = gs[gs["date_str"].isin(wc_2018 | wc_2022)].copy()
    gs_wc = gs_wc[gs_wc["own_goal"] == False]  # excluir autogoles

    # Peso por año: 2022 pesa 2x más que 2018
    gs_wc["year_weight"] = gs_wc["date"].dt.year.map({2018: 0.5, 2022: 1.0})

    wc_stats = gs_wc.groupby(["scorer", "team"]).apply(
        lambda g: pd.Series({
            "wc_goals_weighted": (g["year_weight"]).sum(),
            "wc_goals_2018":     (g["date"].dt.year == 2018).sum(),
            "wc_goals_2022":     (g["date"].dt.year == 2022).sum(),
        })
    ).reset_index().rename(columns={"scorer": "player_name"})

    logger.info(f"  Jugadores con goles en Mundiales 2018/2022: {len(wc_stats)}")
    return wc_stats


def build_player_dataset() -> pd.DataFrame:
    """
    Pipeline completo: carga jugadores activos, calcula forma reciente
    y añade historial de Mundiales.

    Returns:
        DataFrame completo por jugador con todas las métricas
    """
    active  = load_active_players()
    form    = load_player_form(active["player_id"].tolist())
    wc_hist = load_world_cup_history()

    # Unir forma con jugadores activos
    df = active.merge(form, on="player_id", how="left")
    df[["weighted_goals","weighted_assists","weighted_minutes",
        "total_apps","form_score"]] = df[[
        "weighted_goals","weighted_assists","weighted_minutes",
        "total_apps","form_score"]].fillna(0)

    # Unir historial de Mundiales (por nombre — no hay ID común)
    df = df.merge(
        wc_hist[["player_name","team","wc_goals_weighted",
                 "wc_goals_2018","wc_goals_2022"]],
        left_on=["name","country_of_citizenship"],
        right_on=["player_name","team"],
        how="left",
    )
    df[["wc_goals_weighted","wc_goals_2018","wc_goals_2022"]] = \
        df[["wc_goals_weighted","wc_goals_2018","wc_goals_2022"]].fillna(0)

    # Guardar
    out_path = DATA_PROCESSED / "player_dataset.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"Dataset de jugadores guardado: {out_path} ({len(df):,} jugadores)")

    return df


if __name__ == "__main__":
    build_player_dataset()
