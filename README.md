# 🏆 World Cup 2026 Predictor

Predicción del Mundial de Fútbol 2026 usando Machine Learning con modelo de Regresión de Poisson y ensamble XGBoost.

## 📋 Descripción

Este proyecto simula el torneo completo del Mundial 2026 siguiendo el nuevo formato de **48 equipos**, con 12 grupos de 4 equipos y la nueva ronda de dieciseisavos de final.

El modelo predice cuántos goles marcará cada equipo en cada partido, usando una distribución de Poisson para simular resultados. Las variables más importantes son:
- Ranking FIFA (actual e histórico)
- Rendimiento reciente (últimos N partidos)
- Historial de enfrentamientos directos (H2H)
- Ventaja de sede (beneficia a EE. UU., México y Canadá en 2026)
- Fuerza relativa entre equipos basada en puntos FIFA

## 🗂️ Estructura del Proyecto

```
world-cup-2026-predictor/
├── data/
│   ├── raw/                    # Datos originales sin modificar
│   │   ├── results.csv         # Resultados históricos 1872-2026
│   │   ├── matches_1930_2022.csv  # Partidos de Mundiales con detalle
│   │   ├── fifa_ranking-2024-06-20.csv  # Ranking FIFA más reciente
│   │   ├── goalscorers.csv     # Goleadores históricos
│   │   ├── shootouts.csv       # Tandas de penaltis
│   │   ├── world_cup.csv       # Resumen por edición del Mundial
│   │   └── former_names.csv    # Nombres históricos de países
│   └── processed/              # Datos procesados listos para el modelo
├── src/
│   ├── data_preparation.py     # Limpieza y feature engineering
│   ├── model.py                # Entrenamiento del modelo Poisson/XGBoost
│   ├── tournament.py           # Simulación del torneo completo
│   └── utils.py                # Funciones auxiliares
├── notebooks/
│   └── exploratory_analysis.ipynb  # EDA y visualizaciones
├── models/                     # Modelos entrenados serializados
├── results/                    # Predicciones finales
├── requirements.txt
└── main.py                     # Punto de entrada principal
```

## 🚀 Instalación y Uso

### 1. Clonar el repositorio
```bash
git clone https://github.com/tu-usuario/world-cup-2026-predictor.git
cd world-cup-2026-predictor
```

### 2. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 3. Ejecutar la predicción completa
```bash
python main.py
```

Esto ejecuta todo el pipeline:
1. Prepara y procesa los datos
2. Entrena el modelo
3. Simula el torneo completo
4. Guarda los resultados en `results/`

### Ejecución paso a paso
```bash
# Solo preparar datos
python main.py --step prepare

# Solo entrenar modelo
python main.py --step train

# Solo simular torneo
python main.py --step simulate

# Repetir la simulación N veces (Monte Carlo)
python main.py --simulations 10000
```

## 🏟️ Formato del Mundial 2026

| Fase | Detalles |
|------|---------|
| Equipos | 48 |
| Grupos | 12 grupos de 4 equipos |
| Clasificados por grupo | 2 primeros + 8 mejores terceros = 32 equipos |
| Ronda 1/32 | Nueva fase (dieciseisavos) |
| Octavos | 16 equipos |
| Cuartos | 8 equipos |
| Semifinales | 4 equipos |
| Final | 2 equipos |

## 📊 Variables del Modelo (Features)

| Variable | Descripción |
|----------|-------------|
| `fifa_points_home/away` | Puntos FIFA del equipo |
| `fifa_rank_home/away` | Posición en el ranking FIFA |
| `form_home/away` | Promedio de goles marcados (últimos 10 partidos) |
| `defense_home/away` | Promedio de goles recibidos (últimos 10 partidos) |
| `h2h_goals_home/away` | Goles históricos en enfrentamientos directos |
| `is_home_host` | Si el equipo juega en su país sede |
| `rank_diff` | Diferencia de ranking entre equipos |
| `points_ratio` | Ratio de puntos FIFA entre equipos |

## 🤖 Modelo

Se comparan dos enfoques:
1. **Regresión de Poisson** (`scikit-learn`): Modelo estadístico clásico para datos de conteo (goles)
2. **XGBoost Regressor**: Ensamble de árboles para capturar interacciones no lineales

El mejor modelo según validación cruzada se usa para la simulación final.

## 📈 Resultados

Los resultados se guardan en `results/`:
- `group_stage_results.csv`: Resultados de la fase de grupos
- `knockout_bracket.csv`: Eliminatorias completas
- `tournament_winner.txt`: Campeón predicho
- `monte_carlo_probabilities.csv`: Probabilidades de cada equipo si se ejecuta con `--simulations`

## 📁 Fuentes de Datos

- [Kaggle - International Football Results](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017) — `results.csv`, `goalscorers.csv`, `shootouts.csv`
- [Kaggle - FIFA World Cup](https://www.kaggle.com/datasets/abecklas/fifa-world-cup) — `matches_1930_2022.csv`, `world_cup.csv`
- [Kaggle - FIFA Ranking](https://www.kaggle.com/datasets/cashncarry/fifaworldranking) — Archivos `fifa_ranking-*.csv`

## 🧠 Decisiones de Diseño

### ¿Por qué Poisson?
Los goles en fútbol son eventos raros e independientes que siguen aproximadamente una distribución de Poisson. Predecir el número esperado de goles (lambda) de cada equipo es más informativo que predecir directamente victoria/empate/derrota.

### ¿Por qué no predecir directamente el resultado?
Predecir goles permite:
- Simular múltiples escenarios
- Hacer análisis Monte Carlo
- Calcular probabilidades de clasificación
- Manejar empates y penaltis de forma natural

## 📜 Licencia

MIT
