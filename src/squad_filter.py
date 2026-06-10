"""
src/squad_filter.py
Filtra el player_dataset.csv para quedarse solo con los jugadores
convocados al Mundial 2026.

El CSV wc2026_squads.csv debe estar en data/raw/.
Contiene 1221 jugadores de las 48 selecciones con nombres oficiales FIFA.

Como Transfermarkt y FIFA usan nombres distintos (acentos, orden apellido/nombre
en coreano, abreviaturas), este módulo incluye un mapa manual NAME_MAP con
todos los casos conocidos: FIFA_name → TM_name.
"""

import pandas as pd
import unicodedata
from pathlib import Path
from src.utils import DATA_RAW, DATA_PROCESSED, logger

WC_SQUADS_CSV = DATA_RAW / "wc2026_squads.csv"

# ── Mapa manual FIFA → Transfermarkt ─────────────────────────────────────────
# Formato: "Nombre FIFA (equipo)" → "Nombre en TM"
# Solo para casos donde el nombre normalizado (sin acentos) no hace match.
# Verificado contra player_dataset.csv real.
NAME_MAP = {
    # ── España ────────────────────────────────────────────────────────────────
    "Alex Grimaldo":               "Alejandro Grimaldo",

    # ── Alemania ──────────────────────────────────────────────────────────────
    "Pascal Gross":                "Pascal Groß",

    # ── Portugal ──────────────────────────────────────────────────────────────
    "Toti Gomes":                  "Toti",

    # ── Argentina ─────────────────────────────────────────────────────────────
    "Nicolas Gonzalez":            "Nico González",
    "Geronimo Rulli":              "Gerónimo Rulli",
    "Lisandro Martinez":           "Lisandro Martínez",

    # ── Brasil ────────────────────────────────────────────────────────────────
    "Gabriel Magalhaes":           "Gabriel Magalhães",
    "Danilo Luiz":                 "Danilo",        # TM solo guarda "Danilo"
    "Danilo Santos":               "Andrey Santos", # diferente jugador en TM
    "Vinicius Junior":             "Vinicius Júnior",

    # ── Uruguay ───────────────────────────────────────────────────────────────
    "Giogian de Arrascaeta":       "Giorgian de Arrascaeta",
    "Maximiliano Araujo":          "Maxi Araújo",

    # ── Colombia ──────────────────────────────────────────────────────────────
    "Rafael Santos Borre":         "Rafael Borré",
    "Mateus Uribe":                "Matheus Uribe",

    # ── Estados Unidos ────────────────────────────────────────────────────────
    "Tim Weah":                    "Timothy Weah",
    "Gio Reyna":                   "Giovanni Reyna",
    "Alejandro Zendejas":          "Álex Zendejas",

    # ── México ────────────────────────────────────────────────────────────────
    "Guillermo Martinez":          "Emilio Martínez",

    # ── Bélgica ───────────────────────────────────────────────────────────────
    "Branden Mechele":             "Brandon Mechele",

    # ── Croacia ───────────────────────────────────────────────────────────────
    "Petar Suchic":                "Petar Sučić",

    # ── Noruega ───────────────────────────────────────────────────────────────
    "Martin Odegaard":             "Martin Ødegaard",
    "Orjan Nyland":                "Ørjan Nyland",
    "Alexander Sorloth":           "Alexander Sørloth",
    "Torbjorn Heggem":             "Torbjørn Heggem",
    "David Moller Wolfe":          "David Møller Wolfe",
    "Marcus Holmgren Pedersen":    "Marcus Pedersen",

    # ── Turquía ───────────────────────────────────────────────────────────────
    "Ugurcan Cakir":               "Uğurcan Çakır",
    "Altay Bayindir":              "Altay Bayındır",
    "Baris Alper Yilmaz":          "Barış Alper Yılmaz",
    "Ferdi Kadioglu":              "Ferdi Kadıoğlu",
    "Abdulkerim Bardakci":         "Abdülkerim Bardakcı",
    "Kenan Yildiz":                "Kenan Yıldız",

    # ── Japón ─────────────────────────────────────────────────────────────────
    "Yuta Nagatomo":               "Yuto Nagatomo",

    # ── Corea del Sur (orden invertido en TM: apellido-nombre → nombre-apellido)
    "Son Heung-min":               "Heung-min Son",
    "Lee Kang-in":                 "Kang-in Lee",
    "Hwang Hee-chan":              "Hee-chan Hwang",
    "Hwang In-beom":               "In-beom Hwang",
    "Oh Hyeon-gyu":                "Hyeon-gyu Oh",
    "Cho Gue-sung":                "Gue-sung Cho",
    "Paik Seung-ho":               "Seung-ho Paik",
    "Seol Young-woo":              "Young-woo Seol",
    "Lee Dong-gyeong":             "Dong-gyeong Lee",
    "Lee Ki-hyuk":                 "Gi-hyuk Lee",
    "Song Bum-keun":               "Bum-keun Song",
    "Lee Jae-sung":                "Jae-sung Lee",

    # ── Senegal ───────────────────────────────────────────────────────────────
    "Moussa Niakhite":             "Moussa Niakhaté",

    # ── Marruecos ─────────────────────────────────────────────────────────────
    "Munir Mohamedi":              "Munir El Kajoui",
    "Ayoube Amaimouni":            "Ayoube Amaimouni-Echghouyab",

    # ── Egipto ────────────────────────────────────────────────────────────────
    "Mostafa Ziko":                "Mostafa Mohamed",
    "Mostafa Shobeir":             "Oufa Shobeir",
    "Marwan Attia":                "Marwan Hamdi",
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def _norm(name: str) -> str:
    """Normaliza nombre: quita acentos, minúsculas, elimina puntos y guiones."""
    nfkd = unicodedata.normalize('NFKD', str(name))
    ascii_ = nfkd.encode('ascii', 'ignore').decode('ascii')
    return ascii_.lower().strip().replace('.', '').replace('-', ' ')


def _build_lookup_name(fifa_name: str) -> str:
    """Devuelve el nombre TM a buscar: primero mira el mapa, luego normaliza."""
    if fifa_name in NAME_MAP:
        return NAME_MAP[fifa_name]
    # También probar normalizado sin acentos
    norm_key = _norm(fifa_name)
    for k, v in NAME_MAP.items():
        if _norm(k) == norm_key:
            return v
    return fifa_name


# ── Carga convocatorias ───────────────────────────────────────────────────────
def load_wc2026_squad() -> pd.DataFrame:
    if not WC_SQUADS_CSV.exists():
        raise FileNotFoundError(
            f"No se encontró {WC_SQUADS_CSV}\n"
            "Copia wc2026_squads.csv a data/raw/"
        )
    df = pd.read_csv(WC_SQUADS_CSV)
    logger.info(f"Convocatorias WC2026: {len(df)} jugadores de {df['team'].nunique()} selecciones")
    return df


# ── Filtrado principal ────────────────────────────────────────────────────────
def filter_to_wc2026_squad(player_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filtra player_df para conservar solo los convocados al WC2026.

    Matching en orden de preferencia:
      1. Nombre FIFA exacto + equipo
      2. Nombre mapeado (NAME_MAP) + equipo
      3. Nombre normalizado sin acentos + equipo
      4. Solo nombre normalizado (sin equipo) — para cubrir discrepancias de país

    Returns:
        DataFrame filtrado con columna 'position' actualizada desde la convocatoria.
    """
    squads = load_wc2026_squad()

    # Preparar índices de búsqueda en player_df
    pdf = player_df.copy()
    pdf['_name_norm'] = pdf['name'].apply(_norm)
    pdf['_team_norm'] = pdf['country_of_citizenship'].apply(_norm)

    # Para cada convocado, intentar encontrar su fila en player_df
    matched_indices = []
    position_updates = {}

    for _, row in squads.iterrows():
        fifa_name = str(row['name'])
        tm_name   = _build_lookup_name(fifa_name)
        team      = str(row['team'])
        position  = str(row['position'])

        tm_norm   = _norm(tm_name)
        team_norm = _norm(team)

        # Intentos de match (de más a menos estricto)
        match = pd.DataFrame()

        # 1. Nombre mapeado + equipo
        if match.empty:
            match = pdf[(pdf['_name_norm'] == tm_norm) & (pdf['_team_norm'] == team_norm)]

        # 2. Nombre FIFA normalizado + equipo
        if match.empty:
            fifa_norm = _norm(fifa_name)
            match = pdf[(pdf['_name_norm'] == fifa_norm) & (pdf['_team_norm'] == team_norm)]

        # 3. Solo nombre mapeado (sin equipo)
        if match.empty:
            match = pdf[pdf['_name_norm'] == tm_norm]

        # 4. Solo nombre FIFA normalizado (sin equipo)
        if match.empty:
            match = pdf[pdf['_name_norm'] == _norm(fifa_name)]

        if not match.empty:
            idx = match.index[0]
            matched_indices.append(idx)
            position_updates[idx] = position

    # Construir resultado
    matched_indices = list(set(matched_indices))
    filtered = player_df.loc[matched_indices].copy()

    # Actualizar posición con la oficial de la convocatoria
    for idx, pos in position_updates.items():
        if idx in filtered.index:
            filtered.at[idx, 'position'] = pos

    n_found   = len(filtered)
    n_total   = len(squads)
    n_missing = n_total - n_found

    logger.info(f"Filtro WC2026: {n_found}/{n_total} jugadores encontrados en Transfermarkt")
    if n_missing > 0:
        logger.info(f"  {n_missing} convocados sin datos en TM (se añadirán con valores medianos)")

    # Log diagnóstico de quién falta
    found_tm_norms = set(pdf.loc[matched_indices, '_name_norm'].tolist())
    missing_rows = []
    for _, row in squads.iterrows():
        tm_name = _build_lookup_name(str(row['name']))
        if _norm(tm_name) not in found_tm_norms and _norm(str(row['name'])) not in found_tm_norms:
            missing_rows.append(row)
    if missing_rows:
        miss_df = pd.DataFrame(missing_rows)
        logger.debug("  Convocados no encontrados en TM (primeros 30):")
        for _, r in miss_df.head(30).iterrows():
            logger.debug(f"    {r['team']:20s} | {r['name']}")

    return filtered


# Valor mínimo estimado por posición para convocados con valor=0 en TM
WC_MIN_VALUE_BY_POSITION = {
    "Attack":     2_000_000,
    "Midfield":   1_500_000,
    "Defender":   1_000_000,
    "Goalkeeper":   800_000,
}
WC_MIN_VALUE_DEFAULT = 1_000_000


def add_missing_wc_players(player_df: pd.DataFrame,
                            filtered_df: pd.DataFrame) -> pd.DataFrame:
    """
    Dos funciones:
    1. Añade convocados no encontrados en TM con valor mínimo por posición.
    2. Corrige convocados encontrados en TM pero con market_value_in_eur=0.
       Jugadores de ligas africanas/asiáticas poco cubiertas por TM reciben
       un valor mínimo realista para que compitan internamente en su equipo.
       Sin esto, Foster (10M€) acapara el 90% de los goles de Sudáfrica
       porque sus compañeros tienen valor=0.
    """
    squads = load_wc2026_squad()

    # ── Parte 1: añadir jugadores no encontrados en TM ───────────────────────
    found_norms = set(filtered_df['name'].apply(_norm).tolist())
    for fifa_name in squads['name']:
        tm_name = _build_lookup_name(str(fifa_name))
        if _norm(tm_name) in found_norms or _norm(str(fifa_name)) in found_norms:
            found_norms.add(_norm(str(fifa_name)))

    new_rows = []
    for _, row in squads.iterrows():
        if _norm(str(row['name'])) not in found_norms:
            pos     = str(row['position'])
            min_val = WC_MIN_VALUE_BY_POSITION.get(pos, WC_MIN_VALUE_DEFAULT)
            base    = {col: 0 for col in filtered_df.columns}
            base.update({
                'name':                   row['name'],
                'country_of_citizenship': row['team'],
                'position':               pos,
                'market_value_in_eur':    float(min_val),
                'form_score':             0.0,
                'total_apps':             0.0,
                'weighted_goals':         0.0,
                'weighted_assists':       0.0,
                'wc_goals_weighted':      0.0,
                'wc_goals_2018':          0,
                'wc_goals_2022':          0,
            })
            new_rows.append(base)

    if new_rows:
        extra    = pd.DataFrame(new_rows)
        combined = pd.concat([filtered_df, extra], ignore_index=True)
        logger.info(f"  Añadidos {len(new_rows)} jugadores sin datos TM (valor mínimo por posición)")
    else:
        combined = filtered_df.copy()

    # ── Parte 2: corregir value=0 en jugadores ya encontrados en TM ──────────
    zero_mask = combined['market_value_in_eur'] == 0
    n_zeros   = zero_mask.sum()
    if n_zeros > 0:
        def _min_val(pos):
            return float(WC_MIN_VALUE_BY_POSITION.get(str(pos), WC_MIN_VALUE_DEFAULT))
        combined.loc[zero_mask, 'market_value_in_eur'] = (
            combined.loc[zero_mask, 'position'].apply(_min_val)
        )
        logger.info(f"  Corregidos {n_zeros} jugadores con valor=0€ → mínimo por posición")

    return combined


def build_wc2026_player_dataset(include_missing: bool = True) -> pd.DataFrame:
    """
    Pipeline completo:
      1. Carga player_dataset.csv
      2. Filtra a convocados WC2026 con mapa de nombres
      3. Añade convocados sin datos TM con valores medianos
      4. Guarda player_dataset_wc2026.csv

    Returns:
        DataFrame listo para la simulación del torneo.
    """
    player_path = DATA_PROCESSED / "player_dataset.csv"
    if not player_path.exists():
        raise FileNotFoundError(
            "player_dataset.csv no encontrado. Ejecuta --step prepare primero."
        )

    player_df = pd.read_csv(player_path)
    logger.info(f"player_dataset cargado: {len(player_df):,} jugadores")

    filtered = filter_to_wc2026_squad(player_df)

    if include_missing:
        filtered = add_missing_wc_players(player_df, filtered)

    out_path = DATA_PROCESSED / "player_dataset_wc2026.csv"
    filtered.to_csv(out_path, index=False)
    logger.info(f"Dataset WC2026 guardado: {out_path} ({len(filtered):,} jugadores)")

    return filtered


if __name__ == "__main__":
    df = build_wc2026_player_dataset()
    print(f"\nJugadores en dataset WC2026: {len(df)}")
    print(f"Selecciones cubiertas:       {df['country_of_citizenship'].nunique()}")
    print("\nTop 15 por valor de mercado:")
    top = df.nlargest(15, 'market_value_in_eur')[
        ['name','country_of_citizenship','position','market_value_in_eur']
    ]
    for _, r in top.iterrows():
        print(f"  {r['name']:30s} {r['country_of_citizenship']:20s} "
              f"{r['position']:12s} {r['market_value_in_eur']/1e6:.0f}M€")
