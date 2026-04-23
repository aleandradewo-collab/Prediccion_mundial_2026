# 🏆 World Cup 2026 Predictor

Predicción completa del Mundial de Fútbol 2026 usando Machine Learning. El proyecto simula el torneo partido a partido siguiendo el formato oficial de 48 equipos, predice estadísticas individuales de jugadores (máximo goleador, MVP, asistencias) y calcula probabilidades mediante análisis Monte Carlo.

## 📋 ¿Qué hace este proyecto?

- **Predice goles** por partido usando Regresión de Poisson y Gradient Boosting
- **Simula el torneo completo** respetando el formato oficial FIFA 2026 (grupos, bracket, penaltis reales)
- **Calcula probabilidades** de campeonato mediante Monte Carlo (N simulaciones)
- **Predice premios individuales**: máximo goleador, máximo asistente, Balón de Oro
- **Usa datos reales** de Transfermarkt (valores de mercado, forma reciente) y rankings FIFA

## 🗂️ Estructura del Proyecto

```
world-cup-2026-predictor/
├── data/
│   ├── raw/                          # Datos originales sin modificar
│   │   ├── results.csv               # Resultados históricos 1872-2026
│   │   ├── goalscorers.csv           # Goleadores históricos
│   │   ├── shootouts.csv             # Historial de tandas de penaltis
│   │   ├── matches_1930_2022.csv     # Partidos de Mundiales con detalle
│   │   ├── fifa_ranking-2024-06-20.csv  # Ranking FIFA más reciente
│   │   ├── world_cup.csv             # Resumen por edición del Mundial
│   │   ├── former_names.csv          # Nombres históricos de países
│   │   ├── players.csv               # Jugadores Transfermarkt (47k jugadores)
│   │   ├── appearances.csv           # 1.8M apariciones con goles/asist/minutos
│   │   ├── player_valuations.csv     # Historial de valores de mercado
│   │   ├── national_teams.csv        # Datos de selecciones nacionales
│   │   ├── competitions.csv          # Competiciones con sus IDs
│   │   └── Fifa_world_cup_matches.csv # Estadísticas detalladas Mundial 2022
│   └── processed/                    # Datos procesados (generados automáticamente)
│       ├── match_features.csv        # Dataset de entrenamiento
│       ├── team_stats.csv            # Estadísticas por equipo
│       ├── attack_defense_ratings.csv # Ratings Dixon-Coles
│       ├── squad_features.csv        # Features de plantilla (Transfermarkt)
│       └── player_dataset.csv        # Dataset de jugadores con forma y valor
├── src/
│   ├── utils.py                 # Paths, grupos oficiales 2026, pesos de torneos
│   ├── data_preparation.py      # Limpieza, feature engineering y pipeline principal
│   ├── dixon_coles.py           # Ratings de ataque/defensa por equipo
│   ├── model.py                 # Entrenamiento Poisson vs GradientBoosting
│   ├── tournament.py            # Simulación completa del torneo
│   ├── player_data.py           # Carga y procesamiento de datos Transfermarkt
│   ├── squad_strength.py        # Agrega métricas individuales por selección
│   └── player_predictions.py   # Predicciones individuales y premios del torneo
├── notebooks/
│   └── exploratory_analysis.ipynb  # EDA: distribuciones, rankings, correlaciones
├── models/                     # Modelos entrenados (.pkl)
├── results/                    # Predicciones y resultados generados
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

### 3. Añadir los datos de Transfermarkt
Descarga los archivos de Transfermarkt desde Kaggle y colócalos en `data/raw/`:
- `players.csv`, `appearances.csv`, `player_valuations.csv`
- `national_teams.csv`, `competitions.csv`


 `data/raw/appearances.csv` no está incluido en el repositorio por su tamaño (140MB).
> Descárgalo de [Kaggle - Transfermarkt](https://www.kaggle.com/datasets/davidcariboo/player-scores)
> y colócalo en `data/raw/` antes de ejecutar `python main.py --step prepare`.

### 4. Ejecutar el pipeline completo
```bash
python main.py
```

### Ejecución paso a paso
```bash
# Paso 1: preparar datos y calcular features (incluye Dixon-Coles y Transfermarkt)
python main.py --step prepare

# Paso 2: entrenar el modelo (Poisson vs GradientBoosting, elige el mejor)
python main.py --step train

# Paso 3: simular el torneo (1000 simulaciones Monte Carlo por defecto)
python main.py --step simulate

# Solo simulaciones sin reentrenar (más rápido)
python main.py --step simulate --simulations 5000
```

## 🏟️ Formato del Mundial 2026

| Fase | Detalles |
|------|---------|
| Equipos | 48 |
| Grupos | 12 grupos de 4 equipos (A-L) |
| Clasificados | 2 primeros de cada grupo + 8 mejores terceros = 32 |
| Bracket 1/32 | 8 partidos 1º vs 3º · 4 partidos 1º vs 2º · 4 partidos 2º vs 2º |
| Ronda 1/16 | 16 equipos |
| Cuartos | 8 equipos |
| Semifinales | 4 equipos |
| Final | 2 equipos |

**Reglas del bracket**: Nunca equipos del mismo grupo, nunca 1º vs 1º, nunca 3º vs 3º. Los terceros siempre juegan contra primeros.

## 📊 Features del Modelo (37 variables)

### Ranking FIFA
| Variable | Descripción |
|----------|-------------|
| `fifa_points_home/away` | Puntos FIFA actuales |
| `fifa_rank_home/away` | Posición en el ranking |
| `rank_diff` | Diferencia de ranking entre equipos |
| `points_ratio` | Ratio de puntos FIFA |

### Forma reciente
| Variable | Descripción |
|----------|-------------|
| `avg_scored_home/away` | Media de goles marcados (últimos 10 partidos) |
| `avg_conceded_home/away` | Media de goles recibidos |
| `win_rate_home/away` | Tasa de victorias reciente |

### Ratings Dixon-Coles
| Variable | Descripción |
|----------|-------------|
| `attack_rating_home/away` | Potencial ofensivo relativo a la media (>1 = mejor) |
| `defense_rating_home/away` | Solidez defensiva (<1 = mejor defensa) |

> Los ratings Dixon-Coles se calculan con decaimiento temporal (vida media 3 años), ponderación por calidad de rival (rank top-10 vale 2× más que rank 70) y solo con rivales dentro del top 100 del ranking FIFA.

### Transfermarkt — Fuerza de plantilla
| Variable | Descripción |
|----------|-------------|
| `squad_value_home/away` | Valor total de plantilla en M€ |
| `squad_form_home/away` | Forma reciente ponderada de los 23 mejores jugadores |
| `squad_age_home/away` | Edad media de la plantilla |
| `top_scorer_val_home/away` | Valor de mercado del delantero estrella |
| `wc_goals_home/away` | Goles en Mundiales 2018+2022 (2022 pesa 2×) |
| `value_ratio` | Ratio de valor entre plantillas |

### Contexto del partido
| Variable | Descripción |
|----------|-------------|
| `is_neutral` | Si es campo neutral |
| `home_is_host_nation` | Ventaja de sede (EE.UU., México, Canadá) |

## 🤖 Modelo

El pipeline compara dos modelos para predecir los **goles esperados** (lambda de Poisson) de cada equipo:

1. **PoissonRegressor** — modelo estadístico clásico para datos de conteo
2. **GradientBoostingRegressor** — ensamble de árboles que captura interacciones no lineales

La selección se hace por MAE en validación cruzada temporal (`TimeSeriesSplit`), respetando el orden cronológico de los partidos. El modelo ganador suele tener **MAE ≈ 0.93** para goles locales y **≈ 0.80** para visitantes.

### ¿Por qué Poisson?
Los goles en fútbol son eventos raros e independientes. Modelar el número esperado de goles (lambda) en lugar de predecir directamente victoria/empate/derrota permite:
- Simular múltiples escenarios aleatorios
- Hacer análisis Monte Carlo con distribuciones de probabilidad
- Distribuir los goles entre jugadores individuales
- Manejar empates y penaltis de forma natural

## ⚽ Penaltis con datos reales

Las tandas de penaltis no son un 50/50 — se usan las tasas históricas de victoria de cada equipo con dos técnicas:

- **Decaimiento temporal** (vida media 10 años): tandas recientes pesan más
- **Suavizado bayesiano**: evita tasas extremas en equipos con pocos datos

| Equipo | Tasa (suavizada) |
|--------|-----------------|
| Alemania | 0.68 |
| Argentina | 0.67 |
| Países Bajos | 0.27 |
| Inglaterra | 0.57 |
| Suiza | 0.38 |

## 🎯 Predicciones Individuales

En cada simulación, los goles de cada partido se distribuyen entre los jugadores según su probabilidad de marcar, basada en:
- Valor de mercado (40%)
- Forma reciente ponderada por liga (60%)
- Multiplicador por posición (delanteros ×3, centrocampistas ×1.2, defensas ×0.4)

Al final de cada torneo simulado se determinan:
- 🥇 **Máximo goleador**
- 🎯 **Máximo asistente**
- 🏆 **MVP / Balón de Oro** (`goles×2.5 + asistencias×1.5`)

Con Monte Carlo se calculan probabilidades para cada premio.

## 📈 Resultados generados

| Archivo | Contenido |
|---------|-----------|
| `results/group_stage_standings.csv` | Clasificación final de los 12 grupos |
| `results/group_stage_matches.csv` | Todos los partidos de la fase de grupos |
| `results/knockout_bracket.csv` | Resultados de cada ronda eliminatoria |
| `results/tournament_winner.txt` | Campeón predicho |
| `results/player_stats_base.csv` | Estadísticas individuales de la simulación base |
| `results/monte_carlo_probabilities.csv` | P(campeonato) y P(final) por equipo |
| `results/monte_carlo_player_stats.csv` | P(goleador) y P(MVP) por jugador |

## 📁 Fuentes de Datos

| Dataset | Fuente | Contenido |
|---------|--------|-----------|
| `results.csv` | [Kaggle - martj42](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017) | 49k partidos internacionales 1872-2026 |
| `goalscorers.csv` | [Kaggle - martj42](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017) | 47k goles con nombre de jugador |
| `shootouts.csv` | [Kaggle - martj42](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017) | 676 tandas de penaltis históricas |
| `fifa_ranking-*.csv` | [Kaggle - cashncarry](https://www.kaggle.com/datasets/cashncarry/fifaworldranking) | Rankings FIFA históricos |
| `players.csv` + `appearances.csv` | [Transfermarkt / Kaggle](https://www.kaggle.com/datasets/davidcariboo/player-scores) | 47k jugadores, 1.8M apariciones hasta 2026 |

## 🧠 Arquitectura del Pipeline

```
data/raw/
    │
    ├──► data_preparation.py
    │        ├── dixon_coles.py        → attack/defense ratings
    │        ├── player_data.py        → forma y valor por jugador
    │        └── squad_strength.py     → features por selección
    │        └── match_features.csv   (37 features por partido)
    │
    ├──► model.py
    │        └── Poisson vs GBR → mejor modelo por MAE
    │
    └──► tournament.py
             ├── Fase de grupos (72 partidos)
             ├── Bracket oficial FIFA (32→16→8→4→2→1)
             ├── Penaltis con tasas reales
             ├── player_predictions.py → goles por jugador
             └── Monte Carlo (N simulaciones)
```

## 📜 Licencia

MIT
