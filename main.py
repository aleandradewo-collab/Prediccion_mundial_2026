"""
main.py - Punto de entrada del predictor del Mundial 2026.

Uso:
    python main.py                        # Pipeline completo (1000 sims por defecto)
    python main.py --step prepare         # Solo preparar datos
    python main.py --step train           # Solo entrenar modelos
    python main.py --step simulate        # Solo simular torneo (1000 sims)
    python main.py --step simulate --simulations 1000     # Simular con 1000 simulaciones Monte Carlo
    python main.py --simulations 5000     # Cambiar número de simulaciones
    python main.py --simulations 1        # Una sola simulación (bracket único)
    python main.py --help                 # Mostrar ayuda
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.utils import logger, MODELS_DIR, DATA_PROCESSED

DEFAULT_SIMULATIONS = 1 # Para pruebas rápidas, el valor real recomendado es 1000 o más para estabilidad


def parse_args():
    parser = argparse.ArgumentParser(
        description="World Cup 2026 Predictor - Machine Learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Ejemplos:
  python main.py                        # Pipeline completo ({DEFAULT_SIMULATIONS} simulaciones)
  python main.py --step prepare         # Solo limpia datos y crea features
  python main.py --step train           # Solo entrena el modelo
  python main.py --step simulate        # Solo simula el torneo ({DEFAULT_SIMULATIONS} sims)
  python main.py --simulations 5000     # Monte Carlo con 5000 simulaciones
  python main.py --simulations 1        # Un solo bracket (rápido, más aleatorio)
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
        default=DEFAULT_SIMULATIONS,
        help=f"Número de simulaciones Monte Carlo (default: {DEFAULT_SIMULATIONS})",
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
    logger.info(f"Dataset listo: {len(match_features):,} partidos con {match_features.shape[1]} columnas")
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
    logger.info("Modelos entrenados y guardados")
    return model_results


def step_simulate(n_simulations: int):
    logger.info("\n" + "=" * 60)
    logger.info(f"PASO 3: Simulación del Torneo ({n_simulations} simulaciones)")
    logger.info("=" * 60)

    if n_simulations == 1:
        logger.info("AVISO: Con 1 simulación el resultado es muy aleatorio.")
        logger.info("         Usa --simulations 1000 para resultados estables.")

    from src.tournament import run_tournament_simulation
    output = run_tournament_simulation(n_monte_carlo=n_simulations)

    if n_simulations > 1 and "monte_carlo" in output:
        mc = output["monte_carlo"]
        print("\n" + "=" * 60)
        print("  PROBABILIDADES FINALES (Monte Carlo)")
        print("=" * 60)
        print(f"  {'#':<4} {'Equipo':<25} {'% Campeón':>10}  {'% Final':>8}")
        print("  " + "-" * 55)
        for _, row in mc.head(16).iterrows():
            print(f"  {int(row['rank']):<4} {row['team']:<25} "
                  f"{row['p_win_tournament']*100:>9.1f}%  "
                  f"{row['p_reach_final']*100:>7.1f}%")
        print("=" * 60)
        print(f"\n  Basado en {n_simulations} simulaciones completas del torneo.")

    if "monte_carlo_players" in output:
        mp = output["monte_carlo_players"]
        print("\n" + "=" * 60)
        print("  TOP GOLEADORES PROBABLES (Monte Carlo)")
        print("=" * 60)
        print(f"  {'#':<4} {'Jugador':<25} {'Equipo':<20} {'Goles/sim':>9}  {'P(MVP)':>7}")
        print("  " + "-" * 55)
        for _, row in mp.head(10).iterrows():
            print(f"  {int(row['rank_goals']):<4} {row['player']:<25} {row['team']:<20} "
                  f"{row['avg_goals']:>9.2f}  {row['p_mvp']*100:>6.1f}%")
        print("=" * 60)

    return output


def main():
    args = parse_args()

    import numpy as np
    np.random.seed(args.seed)

    print("\n" + "=" * 55)
    print("   WORLD CUP 2026 PREDICTOR — Machine Learning")
    print("=" * 55 + "\n")

    t0 = time.time()

    if args.step in ("prepare", "all"):
        match_features, team_stats, results = step_prepare()
    else:
        match_features = None

    if args.step in ("train", "all"):
        if match_features is None:
            features_path = DATA_PROCESSED / "match_features.csv"
            if not features_path.exists():
                logger.error("No hay datos procesados. Ejecuta primero --step prepare")
                sys.exit(1)
        step_train(match_features)

    if args.step in ("simulate", "all"):
        if not (MODELS_DIR / "model_home.pkl").exists():
            logger.error("No hay modelos entrenados. Ejecuta primero --step train")
            sys.exit(1)
        step_simulate(n_simulations=args.simulations)

    elapsed = time.time() - t0
    print(f"\n  Tiempo total: {elapsed:.1f}s")
    print("  Resultados guardados en: results/\n")


if __name__ == "__main__":
    main()
