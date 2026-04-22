"""
utils.py - Funciones auxiliares para el predictor del Mundial 2026
"""

import os
import json
import logging
from pathlib import Path

# Configuración de paths
ROOT_DIR = Path(__file__).parent.parent
DATA_RAW       = ROOT_DIR / "data" / "raw"
DATA_PROCESSED = ROOT_DIR / "data" / "processed"
MODELS_DIR     = ROOT_DIR / "models"
RESULTS_DIR    = ROOT_DIR / "results"

for d in [DATA_PROCESSED, MODELS_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


# ── Pesos por torneo (para ponderar partidos en el entrenamiento) ─────────────
# Lógica: cuanto más exigente el torneo y mejores los rivales, mayor el peso.
# Torneos regionales de baja exigencia (Gold Cup, COSAFA, etc.) tienen peso
# bajo para evitar que equipos como México o Sudáfrica inflen sus ratings
# acumulando muchos partidos fáciles contra rivales débiles de su confederación.
IMPORTANT_TOURNAMENTS = {
    # Máxima exigencia — los mejores rivales del mundo
    "FIFA World Cup":                       3.0,

    # Alta exigencia — mejores equipos de cada confederación
    "UEFA Euro":                            2.5,
    "Copa América":                         2.5,
    "Africa Cup of Nations":                2.0,
    "AFC Asian Cup":                        2.0,

    # Clasificatorias — rivales mixtos pero contexto competitivo real
    "FIFA World Cup qualification":         1.5,
    "UEFA Euro qualification":              1.2,
    "African Cup of Nations qualification": 1.0,
    "AFC Asian Cup qualification":          1.0,
    "CONCACAF Nations League":              0.9,
    "CONCACAF Nations League qualification":0.7,
    "Gold Cup qualification":               0.7,

    # Torneos de nivel medio
    "UEFA Nations League":                  1.2,
    "Gold Cup":                             0.8,  # rivales débiles de CONCACAF
    "AFF Championship":                     0.7,
    "WAFF Championship":                    0.7,
    "Gulf Cup":                             0.7,
    "EAFF Championship":                    0.7,
    "Arab Cup":                             0.7,
    "SAFF Cup":                             0.6,
    "COSAFA Cup":                           0.5,  # muy regional y débil
    "CECAFA Cup":                           0.5,
    "MSG Prime Minister's Cup":             0.5,
    "Pacific Games":                        0.4,
    "Oceania Nations Cup":                  0.7,

    # Amistosos — valor informativo bajo
    "Friendly":                             0.8,
    "FIFA Series":                          0.8,
    "Intercontinental Cup":                 0.8,
    "King's Cup":                           0.7,
    "Kirin Challenge Cup":                  0.7,
    "Kirin Cup":                            0.7,
    "Baltic Cup":                           0.5,
}

# ── Equipos clasificados al Mundial 2026 (sorteo oficial) ─────────────────────
# Nombres en el formato exacto que usa results.csv
WORLD_CUP_2026_TEAMS = [
    # Grupo A
    "Mexico", "South Africa", "South Korea", "Czech Republic",
    # Grupo B
    "Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland",
    # Grupo C
    "Brazil", "Morocco", "Haiti", "Scotland",
    # Grupo D
    "United States", "Paraguay", "Australia", "Turkey",
    # Grupo E
    "Germany", "Curaçao", "Ivory Coast", "Ecuador",
    # Grupo F
    "Netherlands", "Japan", "Sweden", "Tunisia",
    # Grupo G
    "Belgium", "Egypt", "Iran", "New Zealand",
    # Grupo H
    "Spain", "Cape Verde", "Saudi Arabia", "Uruguay",
    # Grupo I
    "France", "Senegal", "Iraq", "Norway",
    # Grupo J
    "Argentina", "Algeria", "Austria", "Jordan",
    # Grupo K
    "Portugal", "DR Congo", "Uzbekistan", "Colombia",
    # Grupo L
    "England", "Croatia", "Ghana", "Panama",
]

# Países anfitriones (ventaja de localía)
HOST_NATIONS = {"United States", "Mexico", "Canada"}

# Mapeo de nombres alternativos → nombre canónico en results.csv
TEAM_NAME_MAP = {
    "USA":          "United States",
    "EE.UU.":       "United States",
    "Corea":        "South Korea",
    "Chequia":      "Czech Republic",
    "Bosnia":       "Bosnia and Herzegovina",
    "Suiza":        "Switzerland",
    "Marruecos":    "Morocco",
    "Haití":        "Haiti",
    "Escocia":      "Scotland",
    "Turquía":      "Turkey",
    "Alemania":     "Germany",
    "Curazao":      "Curaçao",
    "C. de Marfil": "Ivory Coast",
    "Países Bajos": "Netherlands",
    "Suecia":       "Sweden",
    "Túnez":        "Tunisia",
    "Bélgica":      "Belgium",
    "Egipto":       "Egypt",
    "Irán":         "Iran",
    "N. Zelanda":   "New Zealand",
    "España":       "Spain",
    "Cabo Verde":   "Cape Verde",
    "Arabia S.":    "Saudi Arabia",
    "Francia":      "France",
    "Noruega":      "Norway",
    "Argelia":      "Algeria",
    "Jordania":     "Jordan",
    "RD Congo":     "DR Congo",
    "Uzbekistán":   "Uzbekistan",
    "Inglaterra":   "England",
    "Panamá":       "Panama",
}


def normalize_team_name(name: str) -> str:
    return TEAM_NAME_MAP.get(name, name)


def get_groups() -> dict:
    """
    Grupos oficiales del Mundial 2026 según el sorteo confirmado.
    Nombres en formato results.csv para que el modelo los encuentre.
    """
    return {
        "A": ["Mexico",        "South Africa",           "South Korea",  "Czech Republic"],
        "B": ["Canada",        "Bosnia and Herzegovina", "Qatar",        "Switzerland"],
        "C": ["Brazil",        "Morocco",                "Haiti",        "Scotland"],
        "D": ["United States", "Paraguay",               "Australia",    "Turkey"],
        "E": ["Germany",       "Curaçao",                "Ivory Coast",  "Ecuador"],
        "F": ["Netherlands",   "Japan",                  "Sweden",       "Tunisia"],
        "G": ["Belgium",       "Egypt",                  "Iran",         "New Zealand"],
        "H": ["Spain",         "Cape Verde",             "Saudi Arabia", "Uruguay"],
        "I": ["France",        "Senegal",                "Iraq",         "Norway"],
        "J": ["Argentina",     "Algeria",                "Austria",      "Jordan"],
        "K": ["Portugal",      "DR Congo",               "Uzbekistan",   "Colombia"],
        "L": ["England",       "Croatia",                "Ghana",        "Panama"],
    }


def save_results(data, filename: str, as_json: bool = False):
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
    stages = ["Round of 32", "Round of 16", "Quarter-finals", "Semi-finals", "Final"]
    for stage in stages:
        if stage in bracket:
            print(f"\n{'='*50}")
            print(f"  {stage.upper()}")
            print(f"{'='*50}")
            for match in bracket[stage]:
                home   = match.get("home",       "TBD")
                away   = match.get("away",       "TBD")
                winner = match.get("winner",     "TBD")
                hg     = match.get("home_goals", "-")
                ag     = match.get("away_goals", "-")
                print(f"  {home:25s} {hg} - {ag}  {away:25s}  → {winner}")
