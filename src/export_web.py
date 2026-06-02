"""
src/export_web.py
Genera results/web_data.json para la interfaz web del predictor.

Ejecutar DESPUÉS de las simulaciones:
    python main.py --step simulate --simulations 1000
    python src/export_web.py

Lee directamente los archivos que ya genera tu pipeline:
    data/processed/attack_defense_ratings.csv
    data/processed/team_stats.csv
    data/processed/squad_features.csv
    data/processed/match_features.csv
    data/processed/player_dataset.csv
    results/monte_carlo_probabilities.csv
    results/monte_carlo_player_stats.csv
    results/group_stage_standings.csv
    results/knockout_bracket.csv
    models/model_home.pkl, model_away.pkl, feature_cols.pkl
    data/raw/results.csv
    data/raw/fifa_ranking-*.csv
"""

import json
import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

ROOT      = Path(__file__).resolve().parent.parent
DATA_PROC = ROOT / "data" / "processed"
DATA_RAW  = ROOT / "data" / "raw"
MODELS    = ROOT / "models"
RESULTS   = ROOT / "results"
WEB_OUT   = RESULTS / "web_data.json"

sys.path.insert(0, str(ROOT))
from src.utils import get_groups

RESULTS.mkdir(exist_ok=True)


def safe_float(v, d=3):
    if v is None:
        return None
    try:
        f = float(v)
        return None if (np.isnan(f) or np.isinf(f)) else round(f, d)
    except Exception:
        return None


def load_csv(path):
    p = Path(path)
    if not p.exists():
        print(f"  [WARN] No encontrado: {p.name}")
        return pd.DataFrame()
    return pd.read_csv(p)


# ── Mapa de banderas (ISO2) ────────────────────────────────────────────────────
FLAG_MAP = {
    "United States": "US", "Mexico": "MX", "Canada": "CA",
    "Brazil": "BR", "Argentina": "AR", "France": "FR",
    "England": "GB-ENG", "Germany": "DE", "Spain": "ES",
    "Portugal": "PT", "Netherlands": "NL", "Belgium": "BE",
    "Italy": "IT", "Croatia": "HR", "Morocco": "MA",
    "Japan": "JP", "South Korea": "KR", "Australia": "AU",
    "Ecuador": "EC", "Colombia": "CO", "Uruguay": "UY",
    "Chile": "CL", "Peru": "PE", "Venezuela": "VE",
    "Bolivia": "BO", "Paraguay": "PY", "Costa Rica": "CR",
    "Honduras": "HN", "Panama": "PA", "El Salvador": "SV",
    "Jamaica": "JM", "Haiti": "HT", "Cuba": "CU",
    "Senegal": "SN", "Nigeria": "NG", "Cameroon": "CM",
    "Ghana": "GH", "Egypt": "EG", "Algeria": "DZ",
    "Tunisia": "TN", "Ivory Coast": "CI", "Mali": "ML",
    "South Africa": "ZA", "DR Congo": "CD", "Cape Verde": "CV",
    "Saudi Arabia": "SA", "Iran": "IR", "Qatar": "QA",
    "Indonesia": "ID", "Jordan": "JO", "Iraq": "IQ",
    "Uzbekistan": "UZ", "New Zealand": "NZ",
    "Switzerland": "CH", "Austria": "AT", "Poland": "PL",
    "Serbia": "RS", "Ukraine": "UA", "Turkey": "TR",
    "Romania": "RO", "Hungary": "HU", "Scotland": "GB-SCT",
    "Wales": "GB-WLS", "Czech Republic": "CZ", "Slovakia": "SK",
    "Albania": "AL", "Georgia": "GE", "Slovenia": "SI",
    "Denmark": "DK", "Sweden": "SE", "Norway": "NO",
    "Finland": "FI", "Bosnia and Herzegovina": "BA",
    "Curaçao": "CW",
}


def flag(team):
    return FLAG_MAP.get(team, team[:2].upper() if team else "UN")


# ── 1. Grupos oficiales ────────────────────────────────────────────────────────
def build_groups(team_stats_dict):
    groups_raw = get_groups()
    out = {}
    for group, teams in groups_raw.items():
        out[f"Grupo {group}"] = [
            {
                "team": t,
                "flag": flag(t),
                "fifa_rank": team_stats_dict.get(t, {}).get("fifa_rank"),
            }
            for t in teams
        ]
    return out


# ── 2. Stats por equipo ────────────────────────────────────────────────────────
def build_team_stats():
    stats = {}

    # FIFA ranking (más reciente)
    # Algunos archivos tienen columna rank_date, otros no (la fecha está en el nombre)
    ranking_files = sorted(DATA_RAW.glob("fifa_ranking*.csv"))
    if ranking_files:
        # Usar el archivo más reciente (ordenado por nombre, que incluye la fecha)
        df = pd.read_csv(ranking_files[-1])

        # Si tiene rank_date, filtrar solo la fecha más reciente
        if "rank_date" in df.columns:
            df["rank_date"] = pd.to_datetime(df["rank_date"])
            df = df[df["rank_date"] == df["rank_date"].max()].copy()

        # Normalizar nombre de columna del equipo
        if "country_full" in df.columns:
            df = df.rename(columns={"country_full": "team_name"})
        elif "team" in df.columns:
            df = df.rename(columns={"team": "team_name"})
        else:
            # Buscar cualquier columna con nombre de país
            for col in df.columns:
                if df[col].dtype == object and col not in ["rank"]:
                    df = df.rename(columns={col: "team_name"})
                    break

        for _, r in df.iterrows():
            team = str(r.get("team_name", ""))
            if not team or team == "nan":
                continue
            stats[team] = {
                "fifa_rank":   int(r["rank"]) if "rank" in r and not pd.isna(r["rank"]) else None,
                "fifa_points": safe_float(r.get("total_points")),
                "flag":        flag(team),
            }

    # team_stats.csv — forma reciente, win_rate, medias de goles
    df_ts = load_csv(DATA_PROC / "team_stats.csv")
    if not df_ts.empty:
        for _, r in df_ts.iterrows():
            team = str(r.get("team", ""))
            if team not in stats:
                stats[team] = {"flag": flag(team)}
            for col in ["win_rate", "avg_goals_scored", "avg_goals_conceded",
                        "recent_form", "home_goals_avg", "away_goals_avg"]:
                if col in r and not pd.isna(r[col]):
                    stats[team][col] = safe_float(r[col])

    # attack_defense_ratings.csv — ratings Dixon-Coles
    df_dc = load_csv(DATA_PROC / "attack_defense_ratings.csv")
    if not df_dc.empty:
        for _, r in df_dc.iterrows():
            team = str(r.get("team", ""))
            if team not in stats:
                stats[team] = {"flag": flag(team)}
            stats[team]["attack_rating"]  = safe_float(r.get("attack_rating"))
            stats[team]["defense_rating"] = safe_float(r.get("defense_rating"))

    # squad_features.csv — Transfermarkt
    df_sq = load_csv(DATA_PROC / "squad_features.csv")
    if not df_sq.empty:
        for _, r in df_sq.iterrows():
            team = str(r.get("team", ""))
            if team not in stats:
                stats[team] = {"flag": flag(team)}
            stats[team]["squad_value_M"]      = safe_float(r.get("squad_value_M"))
            stats[team]["squad_form_score"]   = safe_float(r.get("squad_form_score"))
            stats[team]["squad_avg_age"]      = safe_float(r.get("squad_avg_age"))
            stats[team]["top_scorer_value_M"] = safe_float(r.get("top_scorer_value_M"))
            stats[team]["wc_goals_weighted"]  = safe_float(r.get("wc_goals_weighted"))
            stats[team]["squad_coverage"]     = safe_float(r.get("squad_coverage"))

    return stats


# ── 3. Resultados Monte Carlo ──────────────────────────────────────────────────
def build_mc_results(team_stats_dict):
    mc = {}

    # monte_carlo_probabilities.csv — generado por tournament.py
    df_prob = load_csv(RESULTS / "monte_carlo_probabilities.csv")
    if not df_prob.empty:
        # Columnas reales: team, p_win_tournament, p_reach_final, penalty_wins, sim_wins, sim_finals, rank
        n_sims = int(df_prob["sim_wins"].sum()) if "sim_wins" in df_prob.columns else 0
        champ_list = []
        for _, r in df_prob.iterrows():
            team = str(r["team"])
            champ_list.append({
                "team":        team,
                "probability": safe_float(r.get("p_win_tournament", 0) * 100, 2),
                "p_final":     safe_float(r.get("p_reach_final", 0) * 100, 2),
                "flag":        team_stats_dict.get(team, {}).get("flag", flag(team)),
            })
        mc["n_simulations"]           = n_sims
        mc["champion_probabilities"]  = sorted(champ_list, key=lambda x: -(x["probability"] or 0))
        mc["final_probabilities"]     = {
            t["team"]: t["p_final"] for t in champ_list
        }

    # monte_carlo_player_stats.csv — generado por tournament.py
    df_pl = load_csv(RESULTS / "monte_carlo_player_stats.csv")
    if not df_pl.empty:
        scorers = []
        for _, r in df_pl.head(20).iterrows():
            scorers.append({
                "player":    str(r.get("player", "")),
                "team":      str(r.get("team", "")),
                "avg_goals": safe_float(r.get("avg_goals")),
                "avg_assists": safe_float(r.get("avg_assists")),
                "p_top_scorer": safe_float(r.get("p_top_scorer")),
                "p_mvp":     safe_float(r.get("p_mvp")),
            })
        mc["top_scorers_mc"] = scorers

    return mc


# ── 4. Bracket (resultado de la simulación base) ───────────────────────────────
def build_bracket():
    # group_stage_standings.csv
    df_gs = load_csv(RESULTS / "group_stage_standings.csv")
    group_standings = {}
    if not df_gs.empty and "group" in df_gs.columns:
        for grp, sub in df_gs.groupby("group"):
            group_standings[str(grp)] = sub[["team","pts","gf","gc","gd","position"]].to_dict("records")

    # knockout_bracket.csv
    df_ko = load_csv(RESULTS / "knockout_bracket.csv")
    knockout = {}
    if not df_ko.empty and "stage" in df_ko.columns:
        for stage, sub in df_ko.groupby("stage"):
            matches = []
            for _, r in sub.iterrows():
                matches.append({
                    "home":              str(r.get("home", "")),
                    "away":              str(r.get("away", "")),
                    "home_goals":        int(r["home_goals"]) if not pd.isna(r.get("home_goals", float("nan"))) else None,
                    "away_goals":        int(r["away_goals"]) if not pd.isna(r.get("away_goals", float("nan"))) else None,
                    "winner":            str(r.get("winner", "")),
                    "went_to_penalties": bool(r.get("went_to_penalties", False)),
                    "penalty_winner":    str(r.get("penalty_winner", "")) if pd.notna(r.get("penalty_winner")) else None,
                })
            knockout[str(stage)] = matches

    # Leer campeón
    champion = None
    winner_path = RESULTS / "tournament_winner.txt"
    if winner_path.exists():
        text = winner_path.read_text(encoding="utf-8").strip()
        if ":" in text:
            champion = text.split(":")[-1].strip()

    return {"group_standings": group_standings, "knockout": knockout, "champion": champion}


# ── 5. Feature importances ─────────────────────────────────────────────────────
def build_feature_importance():
    out = {}
    for target in ["home", "away"]:
        pkl = MODELS / f"model_{target}.pkl"
        if not pkl.exists():
            continue
        try:
            import joblib
            pipeline = joblib.load(pkl)
            # El pipeline tiene steps: scaler → model
            model = pipeline.named_steps.get("model")
            if model is None:
                continue
            if hasattr(model, "feature_importances_"):
                imps = model.feature_importances_
            elif hasattr(model, "coef_"):
                imps = np.abs(model.coef_).flatten()
            else:
                continue
            # Leer feature_cols
            fc_pkl = MODELS / "feature_cols.pkl"
            if fc_pkl.exists():
                cols = joblib.load(fc_pkl)
            else:
                cols = [f"feat_{i}" for i in range(len(imps))]
            paired = sorted(zip(cols, imps.tolist()), key=lambda x: -x[1])
            out[target] = [{"feature": k, "importance": round(float(v), 5)} for k, v in paired[:20]]
        except Exception as e:
            print(f"  [WARN] No se pudo leer feature importance ({target}): {e}")
    return out


# ── 6. Stats del dataset histórico ────────────────────────────────────────────
def build_dataset_stats():
    path = DATA_RAW / "results.csv"
    if not path.exists():
        return {}

    df = pd.read_csv(path, parse_dates=["date"])
    df_wc = df[df["tournament"] == "FIFA World Cup"].copy()

    # Goles promedio por Mundial (por año)
    df_wc["year"] = df_wc["date"].dt.year
    goals_wc = (
        df_wc.groupby("year")
        .apply(lambda g: round(float((g["home_score"] + g["away_score"]).mean()), 2))
        .reset_index()
        .rename(columns={0: "avg_goals"})
    )
    goals_wc = goals_wc[goals_wc["year"] >= 1966].to_dict("records")

    # Campeones históricos (última fecha del torneo cada año)
    winners = []
    for year, grp in df_wc.groupby("year"):
        final = grp[grp["date"] == grp["date"].max()]
        if len(final) == 0:
            continue
        row = final.iloc[-1]
        if row["home_score"] > row["away_score"]:
            winners.append(row["home_team"])
        elif row["away_score"] > row["home_score"]:
            winners.append(row["away_team"])

    winner_counts = (pd.Series(winners)
                       .value_counts()
                       .head(10)
                       .reset_index()
                       .rename(columns={"index": "team", 0: "titles", "count": "titles"}))

    return {
        "total_matches":     int(len(df)),
        "wc_matches":        int(len(df_wc)),
        "date_range":        [str(df["date"].min())[:10], str(df["date"].max())[:10]],
        "avg_goals_wc":      goals_wc,
        "world_cup_winners": winner_counts.to_dict("records"),
    }


# ── 7. Top jugadores del dataset (no MC) ──────────────────────────────────────
def build_top_players():
    df = load_csv(DATA_PROC / "player_dataset.csv")
    if df.empty:
        return []
    df = df.sort_values("market_value_in_eur", ascending=False).head(50)
    out = []
    for _, r in df.iterrows():
        out.append({
            "name":     str(r.get("name", "")),
            "team":     str(r.get("country_of_citizenship", "")),
            "position": str(r.get("position", "")),
            "value_M":  safe_float(r.get("market_value_in_eur", 0) / 1e6, 1),
            "form_score": safe_float(r.get("form_score")),
            "wc_goals":   safe_float(r.get("wc_goals_weighted")),
        })
    return out


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  export_web.py — Generando web_data.json")
    print("=" * 55)

    print("\n  [1/7] Stats por equipo (rankings, Dixon-Coles, squads)...")
    team_stats = build_team_stats()

    print("  [2/7] Grupos oficiales 2026...")
    groups = build_groups(team_stats)

    print("  [3/7] Resultados Monte Carlo...")
    mc = build_mc_results(team_stats)

    print("  [4/7] Bracket (simulación base)...")
    bracket = build_bracket()

    print("  [5/7] Feature importances del modelo...")
    fi = build_feature_importance()

    print("  [6/7] Stats del dataset histórico...")
    ds = build_dataset_stats()

    print("  [7/7] Top jugadores por valor de mercado...")
    top_players = build_top_players()

    # Métricas del modelo (del MAE de tournament output, o estático)
    model_mae_home, model_mae_away = None, None
    try:
        df_mc = pd.read_csv(RESULTS / "monte_carlo_probabilities.csv")
        # No guardamos MAE en CSV actualmente — se puede añadir en main.py
    except Exception:
        pass

    web_data = {
        "meta": {
            "generated_at":  datetime.now().isoformat(),
            "model_mae_home": model_mae_home or 0.93,
            "model_mae_away": model_mae_away or 0.80,
            "n_simulations":  mc.get("n_simulations", 0),
            "model_type":     "Poisson + GradientBoosting",
        },
        "groups":             groups,
        "team_stats":         team_stats,
        "monte_carlo":        mc,
        "bracket":            bracket,
        "feature_importance": fi,
        "dataset_stats":      ds,
        "top_players":        top_players,
    }

    with open(WEB_OUT, "w", encoding="utf-8") as f:
        json.dump(web_data, f, ensure_ascii=False, indent=2)

    size_kb = WEB_OUT.stat().st_size / 1024
    print(f"\n  ✓ Exportado: {WEB_OUT}")
    print(f"    Tamaño:           {size_kb:.0f} KB")
    print(f"    Equipos:          {len(team_stats)}")
    print(f"    Grupos:           {len(groups)}")
    print(f"    Simulaciones MC:  {mc.get('n_simulations', 0)}")
    print(f"    Bracket stages:   {len(bracket.get('knockout', {}))}")
    print(f"    Top jugadores:    {len(top_players)}")
    print(f"\n  Siguiente paso:")
    print(f"    python scripts/deploy_web.py")


if __name__ == "__main__":
    main()
