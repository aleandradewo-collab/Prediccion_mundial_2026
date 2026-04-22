"""
data_preparation.py - Limpieza de datos y Feature Engineering
para el predictor del Mundial 2026.

Pipeline:
1. Carga resultados históricos de partidos internacionales
2. Incorpora rankings FIFA
3. Construye features por partido: forma reciente, H2H, ranking, ventaja local
4. Genera el dataset de entrenamiento final
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

from src.utils import DATA_RAW, DATA_PROCESSED, normalize_team_name, HOST_NATIONS, logger
from src.dixon_coles import compute_attack_defense_ratings


# ── Parámetros configurables ──────────────────────────────────────────────────
RECENT_MATCHES_WINDOW = 10       # Partidos recientes para calcular forma
MIN_DATE_TRAINING = "2006-01-01" # Solo entrenar con partidos modernos
IMPORTANT_TOURNAMENTS = {        # Peso extra para partidos de mayor importancia
    "FIFA World Cup": 3.0,
    "Copa América": 2.4,
    "UEFA Euro": 2.6,
    "AFC Asian Cup": 1.8,
    "Africa Cup of Nations": 2.1,
    "Gold Cup":              1.8,
    "FIFA World Cup qualification": 1.5,
    "Friendly": 1.0,
}


def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Carga y hace una limpieza básica de los datos crudos.

    Returns:
        results: DataFrame con todos los partidos históricos
        rankings: DataFrame con rankings FIFA (el más reciente disponible)
    """
    logger.info("Cargando datos crudos...")

    # ── Resultados históricos ─────────────────────────────────────────────────
    results = pd.read_csv(DATA_RAW / "results.csv", parse_dates=["date"])
    results = results.dropna(subset=["home_score", "away_score"])
    results["home_score"] = results["home_score"].astype(int)
    results["away_score"] = results["away_score"].astype(int)

    # Filtrar partidos sin sentido (tanteos aberrantes > 20 goles)
    results = results[
        (results["home_score"] <= 20) & (results["away_score"] <= 20)
    ]

    # ── Rankings FIFA ─────────────────────────────────────────────────────────
    # Cargamos todos los archivos y tomamos el snapshot más reciente por equipo
    all_rankings = []
    for rf in sorted(DATA_RAW.glob("fifa_ranking*.csv")):
        tmp = pd.read_csv(rf)
        if "country_full" in tmp.columns:
            tmp = tmp.rename(columns={"country_full": "team", "total_points": "points"})
        if "rank_date" not in tmp.columns:
            date_str = rf.stem.replace("fifa_ranking_", "").replace("fifa_ranking-", "")
            try:
                tmp["rank_date"] = pd.to_datetime(date_str)
            except Exception:
                tmp["rank_date"] = pd.Timestamp("2022-10-06")
        else:
            tmp["rank_date"] = pd.to_datetime(tmp["rank_date"])
        tmp["rank"]   = pd.to_numeric(tmp["rank"],   errors="coerce")
        tmp["points"] = pd.to_numeric(tmp["points"], errors="coerce")
        all_rankings.append(tmp[["team", "rank", "points", "rank_date"]])

    rankings_raw = pd.concat(all_rankings, ignore_index=True).dropna(subset=["team"])
    latest_date  = rankings_raw["rank_date"].max()
    logger.info(f"Usando ranking de fecha: {latest_date.date()}")

    rankings = (
        rankings_raw
        .sort_values("rank_date", ascending=False)
        .drop_duplicates(subset=["team"])
        .reset_index(drop=True)
    )

    logger.info(f"Partidos cargados: {len(results):,}")
    logger.info(f"Equipos en ranking: {len(rankings)}")
    return results, rankings


def build_team_stats(results: pd.DataFrame, rankings: pd.DataFrame) -> pd.DataFrame:
    """
    Construye un DataFrame con estadísticas actuales de cada equipo:
    - Puntos FIFA y ranking
    - Forma reciente (goles marcados/recibidos en últimos N partidos)
    - Historial general

    Returns:
        team_stats: DataFrame indexado por nombre de equipo
    """
    logger.info("Construyendo estadísticas por equipo...")

    # Combinar perspectiva home y away en una sola vista por equipo
    home_view = results[["date", "home_team", "away_team", "home_score", "away_score", "tournament"]].copy()
    home_view.columns = ["date", "team", "opponent", "goals_for", "goals_against", "tournament"]
    home_view["is_home"] = True

    away_view = results[["date", "away_team", "home_team", "away_score", "home_score", "tournament"]].copy()
    away_view.columns = ["date", "team", "opponent", "goals_for", "goals_against", "tournament"]
    away_view["is_home"] = False

    all_matches = pd.concat([home_view, away_view], ignore_index=True)
    all_matches = all_matches.sort_values("date")

    # Filtrar solo los últimos años para calcular forma
    recent_cutoff = pd.Timestamp("2018-01-01")
    recent_matches = all_matches[all_matches["date"] >= recent_cutoff]

    team_stats = {}
    all_teams = set(results["home_team"]) | set(results["away_team"])

    for team in all_teams:
        tm = recent_matches[recent_matches["team"] == team].tail(RECENT_MATCHES_WINDOW)
        if len(tm) == 0:
            continue

        wins = ((tm["goals_for"] > tm["goals_against"])).sum()
        draws = ((tm["goals_for"] == tm["goals_against"])).sum()
        losses = ((tm["goals_for"] < tm["goals_against"])).sum()

        team_stats[team] = {
            "avg_goals_scored": tm["goals_for"].mean(),
            "avg_goals_conceded": tm["goals_against"].mean(),
            "win_rate": wins / len(tm),
            "draw_rate": draws / len(tm),
            "loss_rate": losses / len(tm),
            "total_recent_matches": len(tm),
        }

    team_stats_df = pd.DataFrame(team_stats).T.reset_index()
    team_stats_df.columns = ["team"] + list(team_stats_df.columns[1:])

    # Unir con rankings FIFA
    rankings_slim = rankings[["team", "rank", "points"]].copy()
    rankings_slim.columns = ["team", "fifa_rank", "fifa_points"]
    rankings_slim["fifa_points"] = pd.to_numeric(rankings_slim["fifa_points"], errors="coerce")
    rankings_slim["fifa_rank"] = pd.to_numeric(rankings_slim["fifa_rank"], errors="coerce")

    team_stats_df = team_stats_df.merge(rankings_slim, on="team", how="left")

    # Rellenar equipos sin ranking con valores conservadores
    median_rank = team_stats_df["fifa_rank"].median()
    median_points = team_stats_df["fifa_points"].median()
    team_stats_df["fifa_rank"] = team_stats_df["fifa_rank"].fillna(median_rank * 1.5)
    team_stats_df["fifa_points"] = team_stats_df["fifa_points"].fillna(median_points * 0.5)

    logger.info(f"Estadísticas calculadas para {len(team_stats_df)} equipos")
    return team_stats_df.set_index("team")


def compute_h2h(results: pd.DataFrame, team_a: str, team_b: str, n: int = 10) -> dict:
    """
    Calcula el historial de enfrentamientos directos entre dos equipos.

    Returns:
        dict con goles promedio de cada equipo en los últimos n H2H
    """
    mask = (
        ((results["home_team"] == team_a) & (results["away_team"] == team_b)) |
        ((results["home_team"] == team_b) & (results["away_team"] == team_a))
    )
    h2h = results[mask].tail(n)

    if len(h2h) == 0:
        return {"h2h_goals_a": 1.2, "h2h_goals_b": 1.2, "h2h_matches": 0}

    goals_a = []
    goals_b = []
    for _, row in h2h.iterrows():
        if row["home_team"] == team_a:
            goals_a.append(row["home_score"])
            goals_b.append(row["away_score"])
        else:
            goals_a.append(row["away_score"])
            goals_b.append(row["home_score"])

    return {
        "h2h_goals_a": np.mean(goals_a),
        "h2h_goals_b": np.mean(goals_b),
        "h2h_matches": len(h2h),
    }


def build_match_features(
    results: pd.DataFrame,
    team_stats: pd.DataFrame,
    ratings: pd.DataFrame = None,
    min_date: str = MIN_DATE_TRAINING
) -> pd.DataFrame:
    """
    Construye el dataset de entrenamiento. Cada fila es un partido con
    features de ambos equipos y los goles reales como targets.

    Returns:
        DataFrame listo para entrenar el modelo
    """
    logger.info("Construyendo features de partidos...")

    df = results[results["date"] >= min_date].copy()
    df = df.sort_values("date").reset_index(drop=True)

    # Cargar ranking FIFA para filtrar rivales débiles,
    # igual que hace Dixon-Coles, para que el modelo no aprenda
    # de partidos como México 8-0 Belice
    ranking_files = sorted(DATA_RAW.glob("fifa_ranking*.csv"))
    all_rankings = []
    for rf in ranking_files:
        tmp = pd.read_csv(rf)
        if "country_full" in tmp.columns:
            tmp = tmp.rename(columns={"country_full": "team", "total_points": "points"})
        if "rank_date" not in tmp.columns:
            date_str = rf.stem.replace("fifa_ranking_","").replace("fifa_ranking-","")
            try:    tmp["rank_date"] = pd.to_datetime(date_str)
            except: tmp["rank_date"] = pd.Timestamp("2022-10-06")
        else:
            tmp["rank_date"] = pd.to_datetime(tmp["rank_date"])
        tmp["rank"] = pd.to_numeric(tmp["rank"], errors="coerce")
        all_rankings.append(tmp[["team","rank","rank_date"]])

    latest_ranks = (pd.concat(all_rankings, ignore_index=True)
                      .sort_values("rank_date", ascending=False)
                      .drop_duplicates(subset=["team"])
                      .set_index("team")["rank"])

    MAX_RIVAL_RANK_TRAINING = 100
    EXCLUDE_TOURNAMENTS_TRAINING = {
        "COSAFA Cup", "CECAFA Cup", "CAFA Nations Cup",
        "MSG Prime Minister's Cup", "Pacific Games",
        "Island Games", "CONIFA World Football Cup",
        "CONIFA European Football Cup", "Inter Games",
        "Muratti Vase", "Indian Ocean Island Games",
    }

    before = len(df)
    df = df[~df["tournament"].isin(EXCLUDE_TOURNAMENTS_TRAINING)]
    df = df[
        df["home_team"].map(lambda t: latest_ranks.get(t, 999)) <= MAX_RIVAL_RANK_TRAINING
    ]
    df = df[
        df["away_team"].map(lambda t: latest_ranks.get(t, 999)) <= MAX_RIVAL_RANK_TRAINING
    ].reset_index(drop=True)
    logger.info(f"  Partidos tras filtro de calidad: {len(df):,} (excluidos {before-len(df):,})")

    feature_rows = []

    for _, row in df.iterrows():
        home = row["home_team"]
        away = row["away_team"]

        if home not in team_stats.index or away not in team_stats.index:
            continue

        h = team_stats.loc[home]
        a = team_stats.loc[away]

        is_neutral = row.get("neutral", False)
        home_is_host = home in HOST_NATIONS
        away_is_host = away in HOST_NATIONS

        # Ponderación por importancia del torneo
        tournament_weight = IMPORTANT_TOURNAMENTS.get(row.get("tournament", "Friendly"), 1.0)

        feature_rows.append({
            # Identifiers
            "date": row["date"],
            "home_team": home,
            "away_team": away,
            "tournament": row.get("tournament", "Unknown"),

            # Targets
            "home_goals": row["home_score"],
            "away_goals": row["away_score"],

            # FIFA ranking features
            "fifa_points_home": h.get("fifa_points", 1000),
            "fifa_points_away": a.get("fifa_points", 1000),
            "fifa_rank_home": h.get("fifa_rank", 100),
            "fifa_rank_away": a.get("fifa_rank", 100),
            "rank_diff": h.get("fifa_rank", 100) - a.get("fifa_rank", 100),
            "points_ratio": (
                h.get("fifa_points", 1000) /
                max(a.get("fifa_points", 1000), 1)
            ),

            # Forma reciente
            "avg_scored_home": h.get("avg_goals_scored", 1.2),
            "avg_scored_away": a.get("avg_goals_scored", 1.2),
            "avg_conceded_home": h.get("avg_goals_conceded", 1.2),
            "avg_conceded_away": a.get("avg_goals_conceded", 1.2),
            "win_rate_home": h.get("win_rate", 0.33),
            "win_rate_away": a.get("win_rate", 0.33),

            # Ventaja de sede
            "is_neutral": int(is_neutral),
            "home_is_host_nation": int(home_is_host),
            "away_is_host_nation": int(away_is_host),

            # Peso del partido
            "tournament_weight": tournament_weight,

            # Dixon-Coles ratings (se rellenan con 1.0 si no hay ratings)
            "attack_rating_home":  1.0,
            "attack_rating_away":  1.0,
            "defense_rating_home": 1.0,
            "defense_rating_away":  1.0,
        })

    feature_df = pd.DataFrame(feature_rows)
    # Añadir ratings de ataque/defensa si están disponibles
    if ratings is not None:
        ratings_slim = ratings[["team", "attack_rating", "defense_rating"]].copy()

        # Merge para equipo local
        feature_df = feature_df.merge(
            ratings_slim.rename(columns={
                "team": "home_team",
                "attack_rating":  "attack_rating_home_dc",
                "defense_rating": "defense_rating_home_dc",
            }),
            on="home_team", how="left"
        )
        feature_df["attack_rating_home"]  = feature_df["attack_rating_home_dc"].fillna(1.0)
        feature_df["defense_rating_home"] = feature_df["defense_rating_home_dc"].fillna(1.0)
        feature_df.drop(columns=["attack_rating_home_dc", "defense_rating_home_dc"], inplace=True)

        # Merge para equipo visitante
        feature_df = feature_df.merge(
            ratings_slim.rename(columns={
                "team": "away_team",
                "attack_rating":  "attack_rating_away_dc",
                "defense_rating": "defense_rating_away_dc",
            }),
            on="away_team", how="left"
        )
        feature_df["attack_rating_away"]  = feature_df["attack_rating_away_dc"].fillna(1.0)
        feature_df["defense_rating_away"] = feature_df["defense_rating_away_dc"].fillna(1.0)
        feature_df.drop(columns=["attack_rating_away_dc", "defense_rating_away_dc"], inplace=True)

    logger.info(f"Dataset de entrenamiento: {len(feature_df):,} partidos con {feature_df.shape[1]} columnas")
    return feature_df


def run_preparation_pipeline() -> pd.DataFrame:
    """
    Ejecuta el pipeline completo de preparación de datos.

    Returns:
        match_features: DataFrame listo para entrenar
    """
    results, rankings = load_raw_data()
    team_stats = build_team_stats(results, rankings)

    # Guardar estadísticas de equipos
    team_stats.reset_index().to_csv(DATA_PROCESSED / "team_stats.csv", index=False)
    logger.info("Estadísticas de equipos guardadas en data/processed/team_stats.csv")

    # Calcular ratings de ataque/defensa (Dixon-Coles)
    ratings = compute_attack_defense_ratings(results)
    ratings.to_csv(DATA_PROCESSED / "attack_defense_ratings.csv", index=False)
    logger.info("Ratings guardados en data/processed/attack_defense_ratings.csv")

    match_features = build_match_features(results, team_stats, ratings)
    match_features.to_csv(DATA_PROCESSED / "match_features.csv", index=False)
    logger.info("Features de partidos guardadas en data/processed/match_features.csv")

    return match_features, team_stats, results


if __name__ == "__main__":
    run_preparation_pipeline()
