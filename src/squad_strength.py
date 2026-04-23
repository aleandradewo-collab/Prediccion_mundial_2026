"""
squad_strength.py - Agrega métricas individuales de jugadores
a nivel de selección nacional.

Para cada selección calcula:
  - squad_value_M:        valor de mercado total en M€
  - squad_form_score:     suma ponderada de form_score de los mejores 23
  - squad_avg_age:        edad media de la plantilla
  - top_scorer_value_M:   valor del delantero más caro
  - top_scorer_form:      form_score del mejor delantero
  - wc_goals_weighted:    goles ponderados en Mundiales 2018+2022
  - squad_coverage:       ratio de posiciones cubiertas (0-1)

Gestión de equipos con pocos datos:
  - Si un equipo tiene menos de MIN_PLAYERS jugadores en el dataset,
    se usan valores medianos globales como fallback para no sesgar
    el modelo en su contra (Curaçao, Qatar, etc.)
"""

import numpy as np
import pandas as pd
from datetime import date
from src.utils import DATA_PROCESSED, logger
from src.player_data import build_player_dataset, COUNTRY_NAME_MAP

# Mínimo de jugadores para considerar que los datos son fiables
MIN_PLAYERS = 10

# Jugadores que se toman para calcular la fuerza (los mejores N por valor)
TOP_N_PLAYERS = 23


def _age(dob_str) -> float:
    """Calcula edad actual a partir de fecha de nacimiento."""
    try:
        dob = pd.to_datetime(dob_str)
        today = pd.Timestamp.today()
        return (today - dob).days / 365.25
    except Exception:
        return 26.0  # edad media típica


def build_squad_features(player_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Construye features de fuerza de plantilla por selección.

    Args:
        player_df: resultado de build_player_dataset() — si None, lo recalcula

    Returns:
        DataFrame indexado por nombre de equipo con las features de plantilla
    """
    if player_df is None:
        player_df = build_player_dataset()

    logger.info("Construyendo features de plantilla por selección...")

    # Referencia de edades
    player_df = player_df.copy()
    player_df["age"] = player_df["date_of_birth"].apply(_age)

    all_teams = player_df["country_of_citizenship"].dropna().unique()
    records = []

    # Valores medianos globales para el fallback
    median_value    = player_df["market_value_in_eur"].median()
    median_form     = player_df["form_score"].median()
    median_age      = player_df["age"].median()
    median_wc_goals = player_df["wc_goals_weighted"].median()

    for team in all_teams:
        tm = player_df[player_df["country_of_citizenship"] == team].copy()

        # Tomar los TOP_N_PLAYERS más valiosos (simula convocatoria)
        tm_top = tm.nlargest(TOP_N_PLAYERS, "market_value_in_eur")

        n_players = len(tm_top)
        is_reliable = n_players >= MIN_PLAYERS

        if not is_reliable:
            logger.debug(f"  {team}: solo {n_players} jugadores — usando fallback parcial")

        # Valor total de plantilla
        squad_value = tm_top["market_value_in_eur"].sum() / 1e6

        # Forma media ponderada (los jugadores más valiosos ponderan más)
        if tm_top["market_value_in_eur"].sum() > 0:
            weights = tm_top["market_value_in_eur"] / tm_top["market_value_in_eur"].sum()
            squad_form = (tm_top["form_score"] * weights).sum()
        else:
            squad_form = tm_top["form_score"].mean() if n_players > 0 else median_form

        # Edad media
        squad_avg_age = tm_top["age"].mean() if n_players > 0 else median_age

        # Delanteros estrella
        attackers = tm_top[tm_top["position"] == "Attack"]
        if len(attackers) > 0:
            top_att = attackers.nlargest(1, "market_value_in_eur").iloc[0]
            top_scorer_value = top_att["market_value_in_eur"] / 1e6
            top_scorer_form  = top_att["form_score"]
        else:
            top_scorer_value = squad_value / TOP_N_PLAYERS
            top_scorer_form  = squad_form

        # Goles en Mundiales
        wc_goals = tm["wc_goals_weighted"].sum()

        # Cobertura de posiciones (qué % de posiciones mínimas están cubiertas)
        pos = tm_top["position"].value_counts()
        has_gk  = min(pos.get("Goalkeeper", 0), 2) / 2
        has_def = min(pos.get("Defender",   0), 5) / 5
        has_mid = min(pos.get("Midfield",   0), 6) / 6
        has_att = min(pos.get("Attack",     0), 3) / 3
        coverage = (has_gk + has_def + has_mid + has_att) / 4

        # Fallback para equipos con pocos datos
        if not is_reliable:
            fallback_ratio = n_players / MIN_PLAYERS
            squad_value    = squad_value * fallback_ratio + median_value / 1e6 * (1 - fallback_ratio)
            squad_form     = squad_form  * fallback_ratio + median_form        * (1 - fallback_ratio)

        records.append({
            "team":              team,
            "squad_value_M":     round(squad_value, 2),
            "squad_form_score":  round(squad_form,  4),
            "squad_avg_age":     round(squad_avg_age, 1),
            "top_scorer_value_M": round(top_scorer_value, 2),
            "top_scorer_form":   round(top_scorer_form, 4),
            "wc_goals_weighted": round(wc_goals, 2),
            "squad_coverage":    round(coverage, 3),
            "n_players":         n_players,
            "data_reliable":     is_reliable,
        })

    df = pd.DataFrame(records).set_index("team")

    # Guardar
    out = DATA_PROCESSED / "squad_features.csv"
    df.reset_index().to_csv(out, index=False)
    logger.info(f"Features de plantilla guardadas: {out} ({len(df)} selecciones)")

    # Log top 10 por valor
    logger.info("\n  Top 10 selecciones por valor de plantilla:")
    for team, row in df.nlargest(10, "squad_value_M").iterrows():
        logger.info(f"    {team:<25} {row['squad_value_M']:>8.1f}M€  "
                    f"form={row['squad_form_score']:.3f}  "
                    f"coverage={row['squad_coverage']:.2f}")

    return df


def get_squad_features_for_team(team: str, squad_df: pd.DataFrame) -> dict:
    """
    Devuelve las features de plantilla de un equipo.
    Si no está en el dataset, devuelve la mediana global.
    """
    if team in squad_df.index:
        row = squad_df.loc[team]
        return {
            "squad_value_M":      float(row["squad_value_M"]),
            "squad_form_score":   float(row["squad_form_score"]),
            "squad_avg_age":      float(row["squad_avg_age"]),
            "top_scorer_value_M": float(row["top_scorer_value_M"]),
            "top_scorer_form":    float(row["top_scorer_form"]),
            "wc_goals_weighted":  float(row["wc_goals_weighted"]),
        }
    else:
        # Fallback con medianas globales
        logger.debug(f"  {team}: sin datos de plantilla — usando medianas")
        return {
            "squad_value_M":      float(squad_df["squad_value_M"].median()),
            "squad_form_score":   float(squad_df["squad_form_score"].median()),
            "squad_avg_age":      float(squad_df["squad_avg_age"].median()),
            "top_scorer_value_M": float(squad_df["top_scorer_value_M"].median()),
            "top_scorer_form":    float(squad_df["top_scorer_form"].median()),
            "wc_goals_weighted":  float(squad_df["wc_goals_weighted"].median()),
        }


if __name__ == "__main__":
    build_squad_features()
