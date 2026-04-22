"""
dixon_coles.py - Ratings de ataque y defensa por equipo.

Calcula para cada equipo dos coeficientes:
  - attack_rating  : cuánto marca respecto a la media global (>1 = mejor)
  - defense_rating : cuánto recibe respecto a la media global (<1 = mejor defensa)

Método: ajuste iterativo sobre los últimos N partidos de cada equipo,
ponderando más los partidos recientes y los torneos más importantes.

Estos ratings se combinan con el modelo de Poisson para que una buena
defensa reduzca los goles esperados del rival, no solo el ataque propio.
"""

import numpy as np
import pandas as pd
from src.utils import DATA_RAW, DATA_PROCESSED, IMPORTANT_TOURNAMENTS, logger


# ── Parámetros ────────────────────────────────────────────────────────────────
MIN_DATE          = "2018-01-01"   # Solo partidos recientes para los ratings
MAX_MATCHES       = 60             # Máximo de partidos por equipo (subido al excluir amistosos)
DECAY_WEEKS       = 52 * 3         # Vida media 3 años: equilibrio entre reciente e histórico
HOME_ADVANTAGE    = 1.15           # Factor de localía más conservador
ITERATIONS        = 100            # Iteraciones del ajuste alternado
CONVERGENCE_EPS   = 1e-6           # Criterio de convergencia
# Torneos excluidos del cálculo de ratings.
# Los amistosos (Friendly) NO se excluyen: aportan información aunque sea menor.
# Su impacto ya queda reducido por dos factores combinados:
#   1. peso de torneo bajo (0.8 en IMPORTANT_TOURNAMENTS)
#   2. rival_weight bajo si el rival es débil (rank alto)
# Un amistoso España vs Brasil seguirá aportando. Uno México vs Haití, casi nada.
EXCLUDE_TOURNAMENTS = {
    "COSAFA Cup", "CECAFA Cup", "CAFA Nations Cup",
    "MSG Prime Minister's Cup", "Pacific Games",
    "Island Games", "CONIFA World Football Cup",
    "CONIFA European Football Cup", "Inter Games",
    "Muratti Vase", "Indian Ocean Island Games",
}

# Ranking FIFA máximo del rival para incluir el partido.
# Ignoramos partidos contra equipos muy débiles (rank > umbral).
# Evita que México infle su rating goleando a Belize o Trinidad.
MAX_RIVAL_RANK = 100


def _time_weight(date: pd.Timestamp, reference: pd.Timestamp, decay_weeks: float) -> float:
    """Peso exponencial: partidos más recientes valen más."""
    weeks_ago = max((reference - date).days / 7, 0)
    return np.exp(-np.log(2) * weeks_ago / decay_weeks)


def compute_attack_defense_ratings(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula los ratings de ataque y defensa para todos los equipos.

    Algoritmo (ajuste alternado):
    1. Inicializar attack=1, defense=1 para todos los equipos
    2. Fijar defensa, actualizar ataque: attack_i = media(goles_marcados / (media_global * defense_rival))
    3. Fijar ataque, actualizar defensa: defense_i = media(goles_recibidos / (media_global * attack_rival))
    4. Normalizar para que la media sea 1
    5. Repetir hasta convergencia

    Returns:
        DataFrame con columnas: team, attack_rating, defense_rating
    """
    logger.info("Calculando ratings de ataque/defensa (Dixon-Coles)...")

    reference_date = results_df["date"].max()

    # Filtrar partidos recientes
    df = results_df[results_df["date"] >= MIN_DATE].copy()
    df["date"] = pd.to_datetime(df["date"])

    # Cargar ranking FIFA para filtrar rivales débiles
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

    def get_rank(team):
        return float(latest_ranks.get(team, 999))

    # Excluir torneos de baja calidad
    df = df[~df["tournament"].isin(EXCLUDE_TOURNAMENTS)].copy()
    df = df.dropna(subset=["home_score", "away_score"])

    # Excluir partidos contra rivales demasiado débiles
    before = len(df)
    df = df[
        df["home_team"].map(get_rank).le(MAX_RIVAL_RANK) &
        df["away_team"].map(get_rank).le(MAX_RIVAL_RANK)
    ].copy()
    logger.info(f"  Partidos tras filtros de torneo y ranking rival: {len(df):,} "
                f"(excluidos {before - len(df):,})")

    # Construir vista unificada: cada partido genera dos filas (local y visitante)
    rows = []
    for _, r in df.iterrows():
        tw = _time_weight(r["date"], reference_date, DECAY_WEEKS)

        # Peso por importancia del torneo
        t_weight = IMPORTANT_TOURNAMENTS.get(r.get("tournament", "Friendly"), 1.0)

        # Peso por calidad del rival (strength of schedule):
        # Ganar/perder contra un top-10 aporta más información que contra un top-80.
        # Fórmula: rival_weight = (MAX_RIVAL_RANK - rival_rank + 1) / (MAX_RIVAL_RANK / 2)
        # Con MAX_RIVAL_RANK=100:
        #   rank 1   -> peso 2.00  (máximo)
        #   rank 10  -> peso 1.82
        #   rank 25  -> peso 1.52
        #   rank 50  -> peso 1.02  (neutro, ~media)
        #   rank 100 -> peso 0.02  (mínimo, casi excluido)
        rank_home = get_rank(r["home_team"])
        rank_away = get_rank(r["away_team"])

        rival_weight_home = (MAX_RIVAL_RANK - rank_away + 1) / (MAX_RIVAL_RANK / 2)
        rival_weight_away = (MAX_RIVAL_RANK - rank_home + 1) / (MAX_RIVAL_RANK / 2)

        # Clampear entre 0.1 y 2.0 para evitar pesos extremos
        rival_weight_home = max(0.1, min(rival_weight_home, 2.0))
        rival_weight_away = max(0.1, min(rival_weight_away, 2.0))

        w_base = tw * t_weight

        rows.append({
            "team":           r["home_team"],
            "opponent":       r["away_team"],
            "goals_scored":   r["home_score"],
            "goals_conceded": r["away_score"],
            "is_home":        True,
            "weight":         w_base * rival_weight_home,
        })
        rows.append({
            "team":           r["away_team"],
            "opponent":       r["home_team"],
            "goals_scored":   r["away_score"],
            "goals_conceded": r["home_score"],
            "is_home":        False,
            "weight":         w_base * rival_weight_away,
        })

    matches = pd.DataFrame(rows)

    # Limitar a los últimos MAX_MATCHES por equipo (ya ordenados por peso desc)
    matches = (matches
               .sort_values("weight", ascending=False)
               .groupby("team")
               .head(MAX_MATCHES)
               .reset_index(drop=True))

    # Media global de goles (base de referencia)
    global_mean = np.average(
        matches["goals_scored"],
        weights=matches["weight"]
    )
    logger.info(f"  Media global de goles por partido: {global_mean:.3f}")

    # Inicializar ratings
    teams = sorted(matches["team"].unique())
    attack  = {t: 1.0 for t in teams}
    defense = {t: 1.0 for t in teams}

    # Ajuste alternado
    for iteration in range(ITERATIONS):
        attack_old  = attack.copy()
        defense_old = defense.copy()

        # ── Actualizar ataque ─────────────────────────────────────────────────
        for team in teams:
            tm = matches[matches["team"] == team]
            if len(tm) == 0:
                continue

            numerator   = 0.0
            denominator = 0.0
            for _, row in tm.iterrows():
                opp = row["opponent"]
                home_factor = HOME_ADVANTAGE if row["is_home"] else 1.0
                expected = global_mean * defense.get(opp, 1.0) * home_factor
                numerator   += row["weight"] * row["goals_scored"]
                denominator += row["weight"] * expected

            attack[team] = (numerator / denominator) if denominator > 0 else 1.0

        # ── Actualizar defensa ────────────────────────────────────────────────
        for team in teams:
            tm = matches[matches["team"] == team]
            if len(tm) == 0:
                continue

            numerator   = 0.0
            denominator = 0.0
            for _, row in tm.iterrows():
                opp = row["opponent"]
                home_factor = HOME_ADVANTAGE if not row["is_home"] else 1.0
                expected = global_mean * attack.get(opp, 1.0) * home_factor
                numerator   += row["weight"] * row["goals_conceded"]
                denominator += row["weight"] * expected

            defense[team] = (numerator / denominator) if denominator > 0 else 1.0

        # ── Normalizar (media = 1 en ambos ratings) ───────────────────────────
        atk_mean = np.mean(list(attack.values()))
        def_mean = np.mean(list(defense.values()))
        attack  = {t: v / atk_mean  for t, v in attack.items()}
        defense = {t: v / def_mean  for t, v in defense.items()}

        # ── Convergencia ──────────────────────────────────────────────────────
        atk_diff = max(abs(attack[t] - attack_old[t])  for t in teams)
        def_diff = max(abs(defense[t] - defense_old[t]) for t in teams)
        if max(atk_diff, def_diff) < CONVERGENCE_EPS:
            logger.info(f"  Convergencia en iteracion {iteration + 1}")
            break

    # Construir DataFrame de resultados
    ratings = pd.DataFrame({
        "team":           teams,
        "attack_rating":  [attack[t]  for t in teams],
        "defense_rating": [defense[t] for t in teams],
    })

    # Ordenar por ataque descendente para inspección
    ratings = ratings.sort_values("attack_rating", ascending=False).reset_index(drop=True)

    logger.info(f"  Ratings calculados para {len(ratings)} equipos")
    logger.info("\n  Top 10 por ataque:")
    for _, row in ratings.head(10).iterrows():
        logger.info(f"    {row['team']:25s}  atk={row['attack_rating']:.3f}  def={row['defense_rating']:.3f}")

    logger.info("\n  Top 10 por defensa (menor = mejor):")
    for _, row in ratings.sort_values("defense_rating").head(10).iterrows():
        logger.info(f"    {row['team']:25s}  def={row['defense_rating']:.3f}  atk={row['attack_rating']:.3f}")

    return ratings


def compute_expected_goals(
    home: str,
    away: str,
    ratings: pd.DataFrame,
    global_mean: float = 1.35,
    home_advantage: float = HOME_ADVANTAGE,
    is_neutral: bool = False,
) -> tuple[float, float]:
    """
    Calcula los goles esperados (lambda) usando el modelo Dixon-Coles puro.

    lambda_home = global_mean × attack_home × defense_away × home_adv
    lambda_away = global_mean × attack_away × defense_home

    Args:
        home, away     : nombres de los equipos
        ratings        : DataFrame con attack_rating y defense_rating
        global_mean    : media global de goles por partido
        home_advantage : factor de localía (1.0 en campo neutral)
        is_neutral     : si True, no aplica ventaja de localía

    Returns:
        (lambda_home, lambda_away)
    """
    def get_rating(team, col, default):
        row = ratings[ratings["team"] == team]
        return float(row[col].values[0]) if len(row) > 0 else default

    atk_home = get_rating(home, "attack_rating",  1.0)
    def_home = get_rating(home, "defense_rating", 1.0)
    atk_away = get_rating(away, "attack_rating",  1.0)
    def_away = get_rating(away, "defense_rating", 1.0)

    ha = 1.0 if is_neutral else home_advantage

    lh = global_mean * atk_home * def_away * ha
    la = global_mean * atk_away * def_home

    # Clampear a valores realistas
    lh = max(0.2, min(lh, 6.0))
    la = max(0.2, min(la, 6.0))

    return lh, la


def run_ratings_pipeline(results_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Ejecuta el pipeline completo de ratings y los guarda en data/processed/.

    Returns:
        ratings DataFrame
    """
    if results_df is None:
        results_df = pd.read_csv(DATA_RAW / "results.csv", parse_dates=["date"])

    ratings = compute_attack_defense_ratings(results_df)
    out_path = DATA_PROCESSED / "attack_defense_ratings.csv"
    ratings.to_csv(out_path, index=False)
    logger.info(f"  Ratings guardados en {out_path}")
    return ratings


if __name__ == "__main__":
    run_ratings_pipeline()
