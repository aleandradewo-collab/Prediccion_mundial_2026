"""
main.py - Punto de entrada del predictor del Mundial 2026.

Uso:
    python main.py                        # Pipeline completo
    python main.py --step prepare         # Solo preparar datos
    python main.py --step train           # Solo entrenar modelos
    python main.py --step simulate        # Solo simular torneo
    python main.py --simulations 1000     # Con análisis Monte Carlo
    python main.py --help                 # Mostrar ayuda
"""

import argparse
import sys
import time
from pathlib import Path

# Añadir el root al path para imports relativos
sys.path.insert(0, str(Path(__file__).parent))

from src.utils import logger, MODELS_DIR, DATA_PROCESSED


def parse_args():
    parser = argparse.ArgumentParser(
        description="🏆 World Cup 2026 Predictor - Machine Learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python main.py                       # Ejecuta todo el pipeline
  python main.py --step prepare        # Solo limpia datos y crea features
  python main.py --step train          # Solo entrena el modelo
  python main.py --step simulate       # Solo simula el torneo
  python main.py --simulations 5000    # Análisis Monte Carlo con 5000 sims
        """
    )
    parser.add_argument(
        "--step",
        choices=["prepare", "train", "simulate", "all"],
        default="all",
        help="Paso del pipeline a ejecutar (default: all)",
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=0,
        help="Número de simulaciones Monte Carlo (0 = desactivado)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla aleatoria para reproducibilidad (default: 42)",
    )
    return parser.parse_args()


def step_prepare():
    logger.info("=" * 60)
    logger.info("PASO 1: Preparación de datos y Feature Engineering")
    logger.info("=" * 60)
    from src.data_preparation import run_preparation_pipeline
    match_features, team_stats, results = run_preparation_pipeline()
    logger.info(f"✅ Dataset listo: {len(match_features):,} partidos con {match_features.shape[1]} columnas")
    return match_features, team_stats, results


def step_train(match_features=None):
    logger.info("\n" + "=" * 60)
    logger.info("PASO 2: Entrenamiento del Modelo")
    logger.info("=" * 60)
    from src.model import train, load_features
    if match_features is None:
        match_features = load_features()
    model_results = train(match_features)
    for target, info in model_results.items():
        logger.info(f"  Goles {target}: {info['name']} — MAE={info['mae']:.4f}")
    logger.info("✅ Modelos entrenados y guardados")
    return model_results


def step_simulate(n_monte_carlo: int = 0):
    logger.info("\n" + "=" * 60)
    logger.info("PASO 3: Simulación del Torneo")
    logger.info("=" * 60)
    from src.tournament import run_tournament_simulation
    output = run_tournament_simulation(n_monte_carlo=n_monte_carlo)
    logger.info(f"\n🥇 CAMPEÓN: {output['champion']}")
    return output


def main():
    args = parse_args()

    import numpy as np
    np.random.seed(args.seed)

    print("\n" + "🏆" * 20)
    print("   WORLD CUP 2026 PREDICTOR — Machine Learning")
    print("🏆" * 20 + "\n")

    t0 = time.time()

    if args.step in ("prepare", "all"):
        match_features, team_stats, results = step_prepare()
    else:
        match_features = None

    if args.step in ("train", "all"):
        # Verificar si ya hay datos procesados si solo queremos entrenar
        if match_features is None:
            features_path = DATA_PROCESSED / "match_features.csv"
            if not features_path.exists():
                logger.error("No hay datos procesados. Ejecuta primero --step prepare")
                sys.exit(1)
        step_train(match_features)

    if args.step in ("simulate", "all"):
        # Verificar que hay modelos entrenados
        if not (MODELS_DIR / "model_home.pkl").exists():
            logger.error("No hay modelos entrenados. Ejecuta primero --step train")
            sys.exit(1)
        step_simulate(n_monte_carlo=args.simulations)

    elapsed = time.time() - t0
    print(f"\n⏱️  Tiempo total: {elapsed:.1f}s")
    print("📁 Resultados guardados en: results/")
    print("\n" + "🏆" * 20)


if __name__ == "__main__":
    main()
