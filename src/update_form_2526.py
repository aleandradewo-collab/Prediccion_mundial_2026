"""
src/update_form_2526.py
Actualiza el form_score del player_dataset_wc2026.csv con datos
de la temporada 2025-2026 (5 grandes ligas: PL, La Liga, Bundesliga,
Serie A, Ligue 1).

Uso:
    python src/update_form_2526.py

Para jugadores presentes en el nuevo dataset:
  - form_score se reemplaza por (Gls*2 + Ast*1.5 + Min/90*0.1) * factor
  - total_apps se actualiza con los partidos de la temporada
  - Se mantiene el valor de mercado de Transfermarkt

Para jugadores NO presentes (ligas menores, MLS, Brasil, etc.):
  - Se mantienen los datos originales de Transfermarkt

El factor de escala asegura que los valores sean comparables
con el form_score existente de Transfermarkt (que acumula 2-3 años).
"""

import pandas as pd
import numpy as np
import unicodedata
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import DATA_PROCESSED, DATA_RAW, logger

# ── Paths ──────────────────────────────────────────────────────────────────
NEW_STATS_CSV  = DATA_RAW / "players_data_light-2025_2026.csv"
WC_DATASET_CSV = DATA_PROCESSED / "player_dataset_wc2026.csv"
OUT_CSV        = DATA_PROCESSED / "player_dataset_wc2026.csv"  # overwrite

# Factor de escala: el form_score de TM acumula ~2 años con decaimiento.
# Esta temporada es 1 año reciente sin decaimiento — multiplicamos x1.5
# para que sea comparable en magnitud.
FORM_SCALE_FACTOR = 1.5


def _norm(name: str) -> str:
    n = unicodedata.normalize('NFKD', str(name)).encode('ascii', 'ignore').decode('ascii')
    return n.lower().strip().replace('.', '').replace('-', ' ')


def load_new_stats() -> pd.DataFrame:
    """Carga y limpia el dataset 2025-26."""
    if not NEW_STATS_CSV.exists():
        raise FileNotFoundError(
            f"No se encontró {NEW_STATS_CSV}\n"
            f"Copia players_data_light-2025_2026.csv a data/raw/"
        )

    df = pd.read_csv(NEW_STATS_CSV)
    for col in ['Gls', 'Ast', 'Min', 'MP', '90s']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # form_score_new: misma fórmula que player_data.py
    df['form_score_new'] = (
        df['Gls'] * 2.0 +
        df['Ast'] * 1.5 +
        df['Min'] / 90 * 0.1
    ) * FORM_SCALE_FACTOR

    df['total_apps_new'] = df['MP']
    df['name_norm']      = df['Player'].apply(_norm)

    logger.info(f"Datos 2025-26 cargados: {len(df)} jugadores de las 5 grandes ligas")
    return df[['name_norm', 'Player', 'form_score_new', 'total_apps_new', 'Gls', 'Ast', 'Min', 'Comp']]


def update_player_dataset() -> pd.DataFrame:
    """
    Actualiza player_dataset_wc2026.csv con datos de la temporada 2025-26.
    """
    if not WC_DATASET_CSV.exists():
        raise FileNotFoundError(
            f"No se encontró {WC_DATASET_CSV}\n"
            "Ejecuta primero: python src/squad_filter.py"
        )

    player_df = pd.read_csv(WC_DATASET_CSV)
    new_stats = load_new_stats()

    # Crear lookup por nombre normalizado
    new_lookup = new_stats.set_index('name_norm')

    player_df['name_norm'] = player_df['name'].apply(_norm)

    n_updated = 0
    n_total   = len(player_df)

    for idx, row in player_df.iterrows():
        key = row['name_norm']
        if key in new_lookup.index:
            new_row = new_lookup.loc[key]
            # Si hay varias filas (mismo nombre en varios clubes), usar la de más minutos
            if isinstance(new_row, pd.DataFrame):
                new_row = new_row.nlargest(1, 'Min').iloc[0]

            old_form = player_df.at[idx, 'form_score']
            new_form = float(new_row['form_score_new'])

            # Media ponderada: datos 2025-26 (70%) + histórico TM (30%)
            # Los datos recientes pesan más pero el historial sigue contando.
            # Ej: Kane tuvo gran 2024-25 pero su historial también confirma calidad.
            combined_form = new_form * 0.70 + old_form * 0.30
            player_df.at[idx, 'form_score']  = combined_form
            player_df.at[idx, 'total_apps']  = float(new_row['total_apps_new'])
            n_updated += 1

    # Limpiar columna auxiliar
    player_df.drop(columns=['name_norm'], inplace=True)

    player_df.to_csv(OUT_CSV, index=False)

    logger.info(f"Actualización completada:")
    logger.info(f"  Jugadores actualizados: {n_updated}/{n_total}")
    logger.info(f"  Sin datos 2025-26 (ligas menores): {n_total - n_updated}")

    # Log top jugadores actualizados
    updated_mask = player_df['name'].apply(_norm).isin(new_lookup.index)
    top = player_df[updated_mask].nlargest(10, 'form_score')[
        ['name', 'country_of_citizenship', 'form_score', 'total_apps']
    ]
    logger.info("\n  Top 10 por form_score (post-actualización):")
    for _, r in top.iterrows():
        fr = r['form_score'] / r['total_apps'] if r['total_apps'] > 0 else 0
        logger.info(f"    {r['name']:25s} {r['country_of_citizenship']:15s} "
                    f"form={r['form_score']:.1f} apps={r['total_apps']:.0f} fr={fr:.2f}")

    return player_df


if __name__ == "__main__":
    df = update_player_dataset()
    print(f"\n✓ player_dataset_wc2026.csv actualizado con datos 2025-26")
    print(f"  Jugadores totales: {len(df)}")

    # Mostrar top 15 por form_rate
    df['form_rate'] = df['form_score'] / df['total_apps'].replace(0, np.nan)
    top15 = df.nlargest(15, 'form_rate')[
        ['name', 'country_of_citizenship', 'form_score', 'total_apps', 'form_rate']
    ]
    print("\nTop 15 por form_rate (goles+asist ponderados por partido):")
    for _, r in top15.iterrows():
        print(f"  {r['name']:28s} {r['country_of_citizenship']:15s} "
              f"form={r['form_score']:6.1f} apps={r['total_apps']:3.0f} "
              f"rate={r['form_rate']:.2f}")
