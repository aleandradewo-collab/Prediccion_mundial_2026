"""
model.py - Entrenamiento y evaluación del modelo predictivo de goles.

Compara dos modelos (uno para goles del equipo local, otro para el visitante):
  - GLM Poisson  (PoissonRegressor de sklearn — baseline estadístico clásico)
  - GradientBoostingRegressor (ensamble de sklearn, alternativa a XGBoost)

Selecciona el mejor según MAE en validación cruzada temporal y lo serializa.
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import PoissonRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.pipeline import Pipeline
import logging

from src.utils import DATA_PROCESSED, MODELS_DIR, logger


FEATURE_COLS = [
    "fifa_points_home", "fifa_points_away",
    "fifa_rank_home",   "fifa_rank_away",
    "rank_diff",        "points_ratio",
    "avg_scored_home",  "avg_scored_away",
    "avg_conceded_home","avg_conceded_away",
    "win_rate_home",    "win_rate_away",
    "is_neutral",       "home_is_host_nation", "away_is_host_nation",
]

TARGET_HOME = "home_goals"
TARGET_AWAY = "away_goals"


def load_features() -> pd.DataFrame:
    path = DATA_PROCESSED / "match_features.csv"
    if not path.exists():
        raise FileNotFoundError(
            "No se encontró match_features.csv. "
            "Ejecuta primero data_preparation.py"
        )
    df = pd.read_csv(path, parse_dates=["date"])
    return df.dropna(subset=FEATURE_COLS + [TARGET_HOME, TARGET_AWAY])


def build_poisson_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", PoissonRegressor(alpha=0.1, max_iter=500)),
    ])


def build_gbr_pipeline() -> Pipeline:
    """Gradient Boosting Regressor — no requiere xgboost."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", GradientBoostingRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
            loss="absolute_error",
        )),
    ])


def evaluate_model(pipeline, X, y, cv_splits: int = 5) -> float:
    """MAE medio con TimeSeriesSplit (respeta el orden temporal)."""
    tscv = TimeSeriesSplit(n_splits=cv_splits)
    scores = cross_val_score(
        pipeline, X, y,
        cv=tscv,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )
    return -scores.mean()


def train(df: pd.DataFrame = None) -> dict:
    """
    Entrena Poisson y GBR para goles local y visitante.
    Guarda el mejor de cada par en models/.
    """
    if df is None:
        df = load_features()

    X = df[FEATURE_COLS].values
    y_home = df[TARGET_HOME].values.astype(float)
    y_away = df[TARGET_AWAY].values.astype(float)

    results = {}

    for target_name, y in [("home", y_home), ("away", y_away)]:
        logger.info(f"\nEntrenando modelos para goles del equipo {target_name}...")

        poisson_pipe = build_poisson_pipeline()
        gbr_pipe     = build_gbr_pipeline()

        logger.info("  Evaluando Poisson (CV)...")
        mae_poisson = evaluate_model(poisson_pipe, X, y)
        logger.info(f"  MAE Poisson: {mae_poisson:.4f}")

        logger.info("  Evaluando GradientBoosting (CV)...")
        mae_gbr = evaluate_model(gbr_pipe, X, y)
        logger.info(f"  MAE GBR:     {mae_gbr:.4f}")

        if mae_gbr <= mae_poisson:
            best_pipe, best_mae, best_name = gbr_pipe,     mae_gbr,     "GradientBoosting"
        else:
            best_pipe, best_mae, best_name = poisson_pipe, mae_poisson, "Poisson"

        logger.info(f"  Mejor modelo: {best_name} (MAE={best_mae:.4f})")

        best_pipe.fit(X, y)

        model_path = MODELS_DIR / f"model_{target_name}.pkl"
        joblib.dump(best_pipe, model_path)
        logger.info(f"  Guardado en {model_path}")

        results[target_name] = {"pipeline": best_pipe, "name": best_name, "mae": best_mae}

    joblib.dump(FEATURE_COLS, MODELS_DIR / "feature_cols.pkl")
    logger.info("\n Entrenamiento completado")
    return results


def load_trained_models() -> tuple:
    model_home   = joblib.load(MODELS_DIR / "model_home.pkl")
    model_away   = joblib.load(MODELS_DIR / "model_away.pkl")
    feature_cols = joblib.load(MODELS_DIR / "feature_cols.pkl")
    return model_home, model_away, feature_cols


def predict_goals(model_home, model_away, feature_cols: list, match_features: dict) -> tuple:
    X = np.array([[match_features.get(f, 0.0) for f in feature_cols]])
    lh = float(model_home.predict(X)[0])
    la = float(model_away.predict(X)[0])
    return max(0.2, min(lh, 6.0)), max(0.2, min(la, 6.0))


if __name__ == "__main__":
    train()
