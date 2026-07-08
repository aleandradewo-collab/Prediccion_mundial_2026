# %% [markdown]
# # World Cup 2026 — Model Backtesting
# ## Group Stage Analysis
#
# Este notebook compara las predicciones del modelo ML con los resultados reales
# de la fase de grupos del Mundial 2026.
#
# ### Pasos:
# 1. Rellena `wc2026_real_results.csv` con los marcadores reales
# 2. Rellena `wc2026_real_standings.csv` con la clasificación final de cada grupo
# 3. Ejecuta el notebook completo
#
# Los CSVs están en la misma carpeta que este notebook.

# %% [markdown]
# ## 0. Imports y configuración

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import sys
import warnings
from src.utils import DATA_PROCESSED, logger
warnings.filterwarnings('ignore')

# Añadir el directorio raíz del proyecto al path
sys.path.insert(0, str(DATA_PROCESSED))

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

BLUE  = '#185FA5'
GREEN = '#0F6E56'
RED   = '#B03030'
AMBER = '#BA7517'
GRAY  = '#888780'
LGRAY = '#EEEEEE'

print("Imports OK")

# %% [markdown]
# ## 1. Cargar resultados reales
#
# **Antes de ejecutar esta celda**, rellena `wc2026_real_results.csv`
# con los marcadores reales de los 72 partidos.

# %%
NOTEBOOK_DIR = Path(__file__).parent if '__file__' in dir() else Path('.')
PROJECT_ROOT = NOTEBOOK_DIR # <--- AÑADE ESTA LÍNEA

DATA_PROCESSED_DIR = NOTEBOOK_DIR / 'data' / 'processed'

real_results = pd.read_csv(DATA_PROCESSED_DIR / 'wc2026_real_results.csv')
real_standings = pd.read_csv(DATA_PROCESSED_DIR / 'wc2026_real_standings.csv')

# Verificar que no hay NaN
missing_results  = real_results[real_results['home_goals'].isna()]
missing_standings = real_standings[real_standings['team'].isna() | (real_standings['team'] == '')]

print(f"Total partidos cargados: {len(real_results)}")
print(f"Partidos sin resultado: {len(missing_results)}")
print(f"Posiciones sin equipo: {len(missing_standings)}")
if len(missing_results) > 0:
    print("\nPartidos sin resultado (rellena el CSV):")
    print(missing_results[['home','away','group']].to_string())

# %% [markdown]
# ## 2. Generar predicciones del modelo

# %%
import joblib
import logging
logging.disable(logging.CRITICAL)

try:
    from src.tournament import build_match_features_for_prediction, build_team_stats, load_raw_data
    from src.model import predict_goals

    print("Cargando modelo...")
    results_df, rankings = load_raw_data()
    team_stats = build_team_stats(results_df, rankings)
    if 'team' in team_stats.columns:
        team_stats = team_stats.set_index('team')

    ratings  = pd.read_csv(PROJECT_ROOT / 'data/processed/attack_defense_ratings.csv')
    squad_df = pd.read_csv(PROJECT_ROOT / 'data/processed/squad_features.csv').set_index('team')
    model_h  = joblib.load(PROJECT_ROOT / 'models/model_home.pkl')
    model_a  = joblib.load(PROJECT_ROOT / 'models/model_away.pkl')
    cols     = joblib.load(PROJECT_ROOT / 'models/feature_cols.pkl')

    MODEL_LOADED = True
    print("Modelo cargado correctamente")

except Exception as e:
    print(f"No se pudo cargar el modelo: {e}")
    print("Se usarán lambdas aproximados basados en ratings Dixon-Coles")
    MODEL_LOADED = False

# %%
def get_prediction(home, away):
    """Devuelve (lambda_home, lambda_away) para un partido."""
    if MODEL_LOADED:
        try:
            feats = build_match_features_for_prediction(
                home, away, team_stats, results_df, True,
                ratings=ratings, squad_df=squad_df)
            lh, la = predict_goals(model_h, model_a, cols, feats)
            return round(lh, 3), round(la, 3)
        except Exception as e:
            print(f"  Warning: {home} vs {away} — {e}")
            return 1.2, 1.2
    else:
        # Lambdas aproximados por ratings de ataque conocidos
        ATTACK = {
            'Mexico':1.856,'South Africa':0.953,'South Korea':1.200,'Czech Republic':1.460,
            'Canada':1.570,'Bosnia and Herzegovina':1.440,'Switzerland':1.649,'Qatar':0.856,
            'Brazil':1.773,'Morocco':1.418,'Haiti':0.697,'Scotland':1.510,
            'United States':1.673,'Paraguay':1.500,'Australia':1.580,'Turkey':1.590,
            'Germany':1.730,'Ecuador':1.600,'Ivory Coast':1.480,'Curaçao':0.800,
            'Netherlands':1.758,'Japan':1.660,'Sweden':1.470,'Tunisia':0.900,
            'Belgium':1.735,'Egypt':0.900,'Iran':0.900,'New Zealand':0.700,
            'Spain':1.876,'Cape Verde':0.700,'Saudi Arabia':0.750,'Uruguay':1.673,
            'France':1.875,'Senegal':1.689,'Norway':1.560,'Iraq':0.750,
            'Argentina':1.877,'Algeria':0.900,'Austria':1.595,'Jordan':0.700,
            'Portugal':1.764,'DR Congo':0.850,'Colombia':1.693,'Uzbekistan':0.750,
            'England':1.826,'Croatia':1.717,'Ghana':0.850,'Panama':0.700,
        }
        lh = ATTACK.get(home, 1.2) * (1/ATTACK.get(away, 1.2)) * 1.35
        la = ATTACK.get(away, 1.2) * (1/ATTACK.get(home, 1.2)) * 1.35
        return round(min(max(lh, 0.3), 5.0), 3), round(min(max(la, 0.3), 5.0), 3)


print("Generando predicciones para los 72 partidos...")
preds = []
for _, row in real_results.iterrows():
    lh, la = get_prediction(row['home'], row['away'])
    preds.append({'pred_home': lh, 'pred_away': la})

pred_df = pd.DataFrame(preds)
bt = pd.concat([real_results.reset_index(drop=True), pred_df], axis=1)

# Calcular resultado real y predicho
def outcome(h, a):
    if h > a: return 'home_win'
    elif h == a: return 'draw'
    else: return 'away_win'

def pred_outcome(lh, la, threshold=0.20):
    if lh > la + threshold: return 'home_win'
    elif la > lh + threshold: return 'away_win'
    else: return 'draw'

bt['real_out']  = bt.apply(lambda r: outcome(r['home_goals'], r['away_goals']), axis=1)
bt['pred_out']  = bt.apply(lambda r: pred_outcome(r['pred_home'], r['pred_away']), axis=1)
bt['correct']   = bt['real_out'] == bt['pred_out']
bt['mae_home']  = (bt['pred_home'] - bt['home_goals']).abs()
bt['mae_away']  = (bt['pred_away'] - bt['away_goals']).abs()
bt['total_err'] = bt['mae_home'] + bt['mae_away']

# Guardar
bt.to_csv(NOTEBOOK_DIR / 'backtesting_results.csv', index=False)
print(f"Predicciones generadas. Guardado en backtesting_results.csv")
bt.head(5)

# %% [markdown]
# ## 3. Métricas de rendimiento

# %%
print("="*55)
print("  BACKTESTING — MÉTRICAS PRINCIPALES")
print("="*55)
print(f"  Total partidos analizados: {len(bt)}")
print()
print(f"  Accuracy (W/D/L):       {bt['correct'].mean()*100:.1f}%")
print(f"  MAE goles local:        {bt['mae_home'].mean():.3f}")
print(f"  MAE goles visitante:    {bt['mae_away'].mean():.3f}")
print(f"  MAE combinado:          {((bt['mae_home']+bt['mae_away'])/2).mean():.3f}")
print()
print("  Distribución resultados reales:")
for o, n in bt['real_out'].value_counts().items():
    print(f"    {o:12s}: {n:2d} ({n/len(bt)*100:.1f}%)")
print()
print("  Distribución resultados predichos:")
for o, n in bt['pred_out'].value_counts().items():
    print(f"    {o:12s}: {n:2d} ({n/len(bt)*100:.1f}%)")
print("="*55)
# %% [markdown]
# ## 4. Análisis de clasificación de grupos (Métrica Flexible vs Orden Exacto)

# %%
PRED_GROUPS_TOP3 = {
    'A': ['Mexico', 'South Korea', 'Czech Republic'],
    'B': ['Canada', 'Switzerland', 'Bosnia and Herzegovina'],
    'C': ['Brazil', 'Morocco', 'Scotland'],
    'D': ['United States', 'Paraguay', 'Australia'],
    'E': ['Germany', 'Ivory Coast', 'Ecuador'],
    'F': ['Netherlands', 'Japan', 'Sweden'],
    'G': ['Belgium', 'Iran', 'Egypt'],
    'H': ['Spain', 'Uruguay', 'Saudi Arabia'],
    'I': ['France', 'Senegal', 'Norway'],
    'J': ['Argentina', 'Austria', 'Jordan'],
    'K': ['Portugal', 'Colombia', 'DR Congo'],
    'L': ['England', 'Croatia', 'Ghana'],
}

print(f"{'Grp':>4}  {'Predicho (1º, 2º, 3º)':45s}  {'Real (1º, 2º, 3º)':45s}  {'En Top 3'}  {'Pos. Exactas'}")
print("-"*135)

correct_top3 = 0
correct_exact_order = 0
aciertos_en_top3_por_grupo = []
aciertos_exactos_por_grupo = []
grupos_labels = list('ABCDEFGHIJKL')

for g in grupos_labels:
    grp_real = real_standings[real_standings['group'] == g].sort_values('position')
    real_top3 = grp_real['team'].head(3).tolist()
    pred_top3 = PRED_GROUPS_TOP3.get(g, [])
    
    # 1. Métrica Flexible: ¿Están dentro del bloque de los 3 primeros?
    coincidencias_top3 = len(set(pred_top3) & set(real_top3))
    correct_top3 += coincidencias_top3
    aciertos_en_top3_por_grupo.append(coincidencias_top3)
    
    # 2. Métrica Estricta: ¿Coincide la posición exacta (1º con 1º, 2º con 2º, 3º con 3º)?
    coincidencias_exactas = sum(1 for p, r in zip(pred_top3, real_top3) if p == r)
    correct_exact_order += coincidencias_exactas
    aciertos_exactos_por_grupo.append(coincidencias_exactas)
    
    pred_str = ', '.join(pred_top3)
    real_str = ', '.join(real_top3)
    print(f"  {g}   {pred_str:45s}  {real_str:45s}  {coincidencias_top3}/3 prs.   {coincidencias_exactas}/3 pos.")

print("-"*135)
print(f"RESUMEN DE CLASIFICACIÓN:")
print(f"  -> Equipos que se mantuvieron en el Top 3 (sin importar orden): {correct_top3}/36 ({correct_top3}/36*100:.1f%)")

# %% [markdown]
# ## 5. Partidos más sorprendentes (upsets)

# %%
print("\nTOP 10 PARTIDOS CON MAYOR ERROR DE PREDICCIÓN:")
print("-"*70)
top_errors = bt.nlargest(10, 'total_err')[
    ['home','away','home_goals','away_goals','pred_home','pred_away','total_err','correct']
]
for _, r in top_errors.iterrows():
    ok = 'OK' if r['correct'] else 'WRONG'
    print(f"  {r['home']:20s} {int(r['home_goals'])}-{int(r['away_goals'])} {r['away']:20s}  "
          f"pred={r['pred_home']:.1f}-{r['pred_away']:.1f}  "
          f"err={r['total_err']:.2f}  [{ok}]")

# %% [markdown]
# ## 6. Gráficas Originales

# %%
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('World Cup 2026 — Model Backtesting (Group Stage)', fontsize=16, fontweight='bold', color='#0A2463')

# TL: Scatter
ax = axes[0][0]; ax.set_facecolor('#FAFAFA')
rh = bt['home_goals'].tolist(); ra = bt['away_goals'].tolist()
ph = bt['pred_home'].tolist();  pa = bt['pred_away'].tolist()
ax.scatter(ph, rh, c=BLUE,  s=40, alpha=0.6, label='Home', edgecolors='white', lw=0.5)
ax.scatter(pa, ra, c=GREEN, s=40, alpha=0.6, label='Away', edgecolors='white', lw=0.5)
lm = max(max(rh+ra), max(ph+pa)) + 0.5
ax.plot([0,lm],[0,lm],'--',color='#CCC',lw=1.2,label='Perfect')
ax.set_xlabel('Predicted (lambda)'); ax.set_ylabel('Real goals')
ax.set_xlim(0,lm); ax.set_ylim(0,lm); ax.grid(True, color=LGRAY, lw=0.7); ax.legend(fontsize=9, frameon=False)
ax.set_title('Predicted vs Real Goals', fontweight='bold', color='#0A2463')

# TR: Outcome distribution
ax = axes[0][1]
outcomes = ['home_win','draw','away_win']; lbls = ['Home Win','Draw','Away Win']
rc = [bt['real_out'].value_counts().get(o,0) for o in outcomes]
pc = [bt['pred_out'].value_counts().get(o,0) for o in outcomes]
x = np.arange(3); w = 0.35
b1 = ax.bar(x-w/2, rc, w, color=BLUE, label='Real', zorder=3)
b2 = ax.bar(x+w/2, pc, w, color=GREEN, label='Predicted', alpha=0.85, zorder=3)
ax.set_xticks(x); ax.set_xticklabels(lbls, fontsize=11); ax.legend(frameon=False)
ax.grid(axis='y', color=LGRAY, lw=0.7, zorder=0)
ax.set_title(f'Outcome Distribution — Accuracy: {bt["correct"].mean()*100:.1f}%', fontweight='bold', color='#0A2463')

# BL: Error por grupo
ax = axes[1][0]
grp_mae = bt.groupby('group')[['mae_home','mae_away']].mean()
grp_mae['combined'] = (grp_mae['mae_home'] + grp_mae['mae_away']) / 2
grp_mae = grp_mae.sort_values('combined', ascending=False)
colors_g = [RED if v > grp_mae['combined'].mean() else BLUE for v in grp_mae['combined']]
ax.bar(grp_mae.index, grp_mae['combined'], color=colors_g, zorder=3)
ax.axhline(grp_mae['combined'].mean(), color=AMBER, lw=1.5, linestyle='--', label='Average')
ax.set_title('Prediction Error by Group (MAE)', fontweight='bold', color='#0A2463')

# BR: Error distribution histogram
ax = axes[1][1]
all_errors = list(bt['mae_home']) + list(bt['mae_away'])
ax.hist(all_errors, bins=15, color=BLUE, alpha=0.8, edgecolor='white', zorder=3)
ax.set_title('Goal Prediction Error Distribution', fontweight='bold', color='#0A2463')

plt.tight_layout()
plt.savefig(NOTEBOOK_DIR / 'backtesting_charts.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nGráficas base guardadas en backtesting_charts.png")

# %% [markdown]
# ## 7. Análisis de Brier Score

# %%
from scipy.stats import poisson

def poisson_probs(lh, la):
    max_goals = 10
    p_home_win = 0; p_draw = 0; p_away_win = 0
    for h in range(max_goals+1):
        for a in range(max_goals+1):
            p = poisson.pmf(h, lh) * poisson.pmf(a, la)
            if h > a: p_home_win += p
            elif h == a: p_draw += p
            else: p_away_win += p
    return p_home_win, p_draw, p_away_win

brier_scores = []
for _, r in bt.iterrows():
    ph, pd_val, pa = poisson_probs(r['pred_home'], r['pred_away'])
    probs = {'home_win': ph, 'draw': pd_val, 'away_win': pa}
    actual = {o: 1 if r['real_out'] == o else 0 for o in ['home_win','draw','away_win']}
    bs = sum((probs[o] - actual[o])**2 for o in ['home_win','draw','away_win']) / 3
    brier_scores.append(bs)

mean_bs = np.mean(brier_scores)

# %% [markdown]
# ## 8. Resumen final

# %%
print("\n" + "="*60)
print("  RESUMEN BACKTESTING — WORLD CUP 2026 FASE DE GRUPOS")
print("="*60)
print(f"  Partidos analizados:              {len(bt)}")
print(f"  Accuracy Partidos (W/D/L):        {bt['correct'].mean()*100:.1f}%")
print(f"  MAE goles local:                  {bt['mae_home'].mean():.3f}")
print(f"  MAE goles visitante:              {bt['mae_away'].mean():.3f}")
print(f"  Brier Score (Calibración):        {mean_bs:.4f}")
print(f"  Aciertos en bloque Top 3:         {correct_top3}/36 ({correct_top3/36*100:.1f}%)")
print(f"  Aciertos en ORDEN EXACTO (1º2º3º): {correct_exact_order}/36 ({correct_exact_order}/36*100:.1f%)")
print("="*60)

# %% [markdown]
# ## 9. Imagen 1: Rendimiento General de Partidos (prediction_analysis.png)

# %%
print("\nGenerando Imagen 1: `prediction_analysis.png`...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Análisis de Goles y Resultados (Fase de Grupos)', fontsize=14, fontweight='bold', color='#0A2463')

# Tarta de aciertos generales
correct_count = bt['correct'].sum()
wrong_count = len(bt) - correct_count
ax1.pie([correct_count, wrong_count], labels=['Aciertos Signo (OK)', 'Errores Signo (WRONG)'], autopct='%1.1f%%', startangle=90, colors=[GREEN, RED], wedgeprops={'edgecolor': 'white', 'linewidth': 2, 'alpha': 0.85})
ax1.set_title('Porcentaje Total de Aciertos (Signo 1X2)', fontweight='bold', color='#0A2463')

# Top 5 Upsets
top_5_errors = bt.nlargest(5, 'total_err')
labels_partidos = [f"{r['home']}\nvs\n{r['away']}" for _, r in top_5_errors.iterrows()]
bars5 = ax2.bar(labels_partidos, top_5_errors['total_err'], color=AMBER, alpha=0.85, edgecolor='black', lw=0.5)
ax2.set_ylabel('Error acumulado (Goles)')
ax2.set_title('Top 5 Partidos con Mayor Desviación de Goles (Upsets)', fontweight='bold', color='#0A2463')
ax2.grid(axis='y', color=LGRAY, lw=0.7)
for bar in bars5:
    ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05, f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
NUEVO_PNG = NOTEBOOK_DIR / 'prediction_analysis.png'
plt.savefig(NUEVO_PNG, dpi=150, bbox_inches='tight')
plt.close()

# %% [markdown]
# ## 10. Imagen 2: Nueva gráfica comparativa de Orden de Grupos (group_order_analysis.png)

# %%
print("Generando Imagen 2: `group_order_analysis.png` (Métrica Estricta vs Flexible)...")

fig, ax = plt.subplots(figsize=(15, 6))
x = np.arange(len(grupos_labels))
width = 0.35

# Barras comparativas por grupo
rects1 = ax.bar(x - width/2, aciertos_en_top3_por_grupo, width, label='En bloque Top 3 (Flexible)', color=BLUE, alpha=0.85, edgecolor='black', lw=0.5)
rects2 = ax.bar(x + width/2, aciertos_exactos_por_grupo, width, label='En Posición Exacta (Estricto)', color=GREEN, alpha=0.85, edgecolor='black', lw=0.5)

ax.set_title('Precisión por Grupos: Equipos Clasificados vs Orden Exacto de Posición', fontsize=14, fontweight='bold', color='#0A2463')
ax.set_xlabel('Grupos del Mundial')
ax.set_ylabel('Número de equipos acertados (Máx 3)')
ax.set_xticks(x)
ax.set_xticklabels(grupos_labels)
ax.set_ylim(0, 3.8)
ax.set_yticks([0, 1, 2, 3])
ax.grid(axis='y', color=LGRAY, lw=0.7, zorder=0)
ax.legend(frameon=False, loc='upper right')

# Añadir etiquetas con los valores encima de las barras
for rect in rects1:
    ax.text(rect.get_x() + rect.get_width()/2., rect.get_height() + 0.05, f'{int(rect.get_height())}', ha='center', va='bottom', fontsize=8, color='#333')
for rect in rects2:
    ax.text(rect.get_x() + rect.get_width()/2., rect.get_height() + 0.05, f'{int(rect.get_height())}', ha='center', va='bottom', fontsize=8, color='darkgreen', fontweight='bold')

plt.tight_layout()
GRUPOS_PNG = NOTEBOOK_DIR / 'group_order_analysis.png'
plt.savefig(GRUPOS_PNG, dpi=150, bbox_inches='tight')
plt.close()

print(f"¡Análisis completo! Las dos imágenes se guardaron perfectamente sin colgar la consola:\n  -> {NUEVO_PNG}\n  -> {GRUPOS_PNG}")