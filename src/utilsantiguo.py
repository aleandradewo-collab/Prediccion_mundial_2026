"""
utils.py - Funciones auxiliares para el predictor del Mundial 2026
"""

import os
import json
import logging
from pathlib import Path

# Configuración de paths
ROOT_DIR = Path(__file__).parent.parent
DATA_RAW = ROOT_DIR / "data" / "raw"
DATA_PROCESSED = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"

# Crear directorios si no existen
for d in [DATA_PROCESSED, MODELS_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


# ── Equipos clasificados al Mundial 2026 ──────────────────────────────────────
# Basado en el ranking FIFA de junio 2024 + clasificatorias (estimación)
# Se actualizará con la clasificación real una vez confirmada.
WORLD_CUP_2026_TEAMS = [
    # UEFA (16 equipos)
    "Spain", "France", "England", "Portugal", "Netherlands",
    "Belgium", "Germany", "Croatia", "Italy", "Switzerland",
    "Austria", "Denmark", "Serbia", "Poland", "Hungary", "Scotland",

    # CONMEBOL (6 equipos)
    "Argentina", "Brazil", "Colombia", "Uruguay", "Ecuador", "Venezuela",

    # CONCACAF (6 equipos — incluye los 3 anfitriones)
    "USA", "Mexico", "Canada", "Costa Rica", "Panama", "Honduras",

    # CAF (9 equipos)
    "Morocco", "Senegal", "Nigeria", "Egypt", "Ivory Coast",
    "Cameroon", "Ghana", "South Africa", "Tunisia",

    # AFC (8 equipos)
    "Japan", "South Korea", "IR Iran", "Australia",
    "Saudi Arabia", "Qatar", "Uzbekistan", "Iraq",

    # OFC (1 equipo)
    "New Zealand",

    # Repesca (2 equipos)
    "Ukraine", "Venezuela",
]

# Aseguramos exactamente 48 equipos únicos
WORLD_CUP_2026_TEAMS = list(dict.fromkeys(WORLD_CUP_2026_TEAMS))[:48]

# Países anfitriones (ventaja de localía)
HOST_NATIONS = {"USA", "United States", "Mexico", "Canada"}

# Mapeo de nombres alternativos hacia el nombre canónico usado en results.csv
TEAM_NAME_MAP = {
    "United States": "USA",
    "IR Iran": "Iran",
    "South Korea": "Korea Republic",
    "Ivory Coast": "Côte d'Ivoire",
    "New Zealand": "New Zealand",
    "Saudi Arabia": "Saudi Arabia",
}


def normalize_team_name(name: str) -> str:
    """Normaliza el nombre de un equipo al formato estándar del dataset."""
    return TEAM_NAME_MAP.get(name, name)


def get_groups() -> dict[str, list[str]]:
    """
    Genera los 12 grupos del Mundial 2026 (4 equipos por grupo).
    En un proyecto real, estos grupos vendrían del sorteo oficial.
    Aquí los distribuimos de forma representativa por confederación y ranking.
    """
    teams = WORLD_CUP_2026_TEAMS.copy()

    # Distribución aproximada respetando las restricciones de la FIFA
    # (equipos de la misma confederación no comparten grupo salvo CONCACAF/UEFA)
    groups = {
        "A": ["USA", "England", "Morocco", "Japan"],
        "B": ["Mexico", "Spain", "Senegal", "South Korea"],
        "C": ["Canada", "France", "Nigeria", "Australia"],
        "D": ["Argentina", "Portugal", "Egypt", "Switzerland"],
        "E": ["Brazil", "Germany", "Ivory Coast", "IR Iran"],
        "F": ["France", "Netherlands", "Ghana", "Saudi Arabia"],
        "G": ["Colombia", "Croatia", "Cameroon", "New Zealand"],
        "H": ["Uruguay", "Belgium", "South Africa", "Qatar"],
        "I": ["Ecuador", "Italy", "Tunisia", "Uzbekistan"],
        "J": ["Costa Rica", "Denmark", "Serbia", "Iraq"],
        "K": ["Panama", "Austria", "Poland", "Venezuela"],
        "L": ["Honduras", "Hungary", "Scotland", "Ukraine"],
    }

    # Eliminar duplicados si Francia aparece dos veces (ajuste manual del ejemplo)
    # En producción, usar el sorteo oficial
    seen = set()
    for group_name, group_teams in groups.items():
        groups[group_name] = []
        for t in group_teams:
            if t not in seen:
                groups[group_name].append(t)
                seen.add(t)

    return groups


def save_results(data, filename: str, as_json: bool = False):
    """Guarda resultados en el directorio de resultados."""
    path = RESULTS_DIR / filename
    if as_json:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    else:
        import pandas as pd
        if isinstance(data, pd.DataFrame):
            data.to_csv(path, index=False)
        else:
            with open(path, "w", encoding="utf-8") as f:
                f.write(str(data))
    logger.info(f"Guardado: {path}")


def print_bracket(bracket: dict):
    """Imprime el bracket del torneo de forma legible."""
    stages = ["Round of 32", "Round of 16", "Quarter-finals", "Semi-finals", "Final"]
    for stage in stages:
        if stage in bracket:
            print(f"\n{'='*50}")
            print(f"  {stage.upper()}")
            print(f"{'='*50}")
            for match in bracket[stage]:
                home = match.get("home", "TBD")
                away = match.get("away", "TBD")
                winner = match.get("winner", "TBD")
                hg = match.get("home_goals", "-")
                ag = match.get("away_goals", "-")
                print(f"  {home:20s} {hg} - {ag}  {away:20s}  → {winner}")
