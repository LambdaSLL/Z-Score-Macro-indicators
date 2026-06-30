"""
zscore_indicators.py
---------------------
Construye un indicador Z-Score compuesto de desviación macro en DOS
niveles de estandarización, siguiendo la misma lógica que el Chicago Fed
National Activity Index (CFNAI) y el Aruoba-Diebold-Scotti (ADS) Index:

  Nivel 1: cada indicador individual se transforma (según la transformación
           que las pruebas ADF/KPSS determinen como la más adecuada, ver
           stationarity.py) y se estandariza a su propio Z-Score.

  Nivel 2: los Z-Scores individuales de una categoría (crecimiento,
           inflación, empleo) se combinan en un promedio ponderado (pesos
           que el PM define y que deben sumar 1 dentro de cada categoría),
           y ESE PROMEDIO PONDERADO SE VUELVE A ESTANDARIZAR — porque un
           promedio ponderado de Z-Scores correlacionados no tiene
           automáticamente desviación estándar 1. Así se obtiene el
           Z-Score de categoría.

  Nivel 3: los 3 Z-Scores de categoría se combinan con un peso libre entre
           ellos (p. ej. 0.4/0.4/0.2) y ESE COMPUESTO TAMBIÉN SE
           RE-ESTANDARIZA, dando el Z-Score maestro final.

Cada nivel de combinación incluye una función de DESCOMPOSICIÓN que
reparte el Z-Score final entre sus componentes (cuánto aportó cada
indicador / cada categoría a la desviación observada), para poder explicar
"por qué" el índice está donde está.

Referencias metodológicas:
- Stock, J. H., & Watson, M. W. (1999). Forecasting inflation. JME 44(2).
- McCracken, M. W., & Ng, S. (2016). FRED-MD: A monthly database for
  macroeconomic research. JBES 34(4), 574-589. (selección de
  transformación por serie, ver stationarity.py)
- Aruoba, S. B., Diebold, F. X., & Scotti, C. (2009). Real-Time
  Measurement of Business Conditions. JBES 27(4), 417-427.
- Federal Reserve Bank of Chicago, "Background on the CFNAI": doble
  estandarización (componentes y compuesto final ambos a media 0 / sd 1).

Diferencia deliberada con el CFNAI: los pesos NO se estiman vía PCA, sino
que quedan 100% en manos del usuario (PM).
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from src.stationarity import select_best_transform, apply_candidate_transform


# ---------------------------------------------------------------------------
# Definición de indicadores por categoría
# ---------------------------------------------------------------------------
# indicator_type: "rate" (ya en %, p. ej. desempleo, breakevens) o "index"
#                 (nivel con tendencia, p. ej. PIB, producción, nóminas).
#                 Determina qué transformaciones candidatas se prueban
#                 (ver stationarity.py: RATE_CANDIDATES / INDEX_CANDIDATES).
# sign: 1 si "más alto = mejor" para esa categoría, -1 si "más alto = peor"
#       (p. ej. desempleo o solicitudes de desempleo deben invertirse).

GROWTH_INDICATORS = {
    "GDPC1":   {"indicator_type": "index", "sign": 1, "label": "PIB real"},
    "INDPRO":  {"indicator_type": "index", "sign": 1, "label": "Producción industrial"},
    "RSAFS":   {"indicator_type": "index", "sign": 1, "label": "Ventas minoristas"},
    "CUMFNS":  {"indicator_type": "rate",  "sign": 1, "label": "Utilización de capacidad"},
}

INFLATION_INDICATORS = {
    "CPIAUCSL_YOY": {"indicator_type": "rate", "sign": 1, "label": "CPI total (YoY %)"},
    "PCEPI_YOY":    {"indicator_type": "rate", "sign": 1, "label": "PCE (YoY %)"},
    "T5YIE":        {"indicator_type": "rate", "sign": 1, "label": "Breakeven inflación 5y"},
    "T10YIE":       {"indicator_type": "rate", "sign": 1, "label": "Breakeven inflación 10y"},
    "MICH":         {"indicator_type": "rate", "sign": 1, "label": "Expectativas U. Michigan 1y"},
}

EMPLOYMENT_INDICATORS = {
    "PAYEMS":  {"indicator_type": "index", "sign": 1,  "label": "Nóminas no agrícolas"},
    "UNRATE":  {"indicator_type": "rate",  "sign": -1, "label": "Tasa de desempleo (invertida)"},
    "ICSA":    {"indicator_type": "index", "sign": -1, "label": "Solicitudes desempleo semanal (invertida)"},
    "AHETPI":  {"indicator_type": "index", "sign": 1,  "label": "Salario promedio por hora"},
}

CATEGORY_DEFINITIONS = {
    "growth": GROWTH_INDICATORS,
    "inflation": INFLATION_INDICATORS,
    "employment": EMPLOYMENT_INDICATORS,
}

TRANSFORM_LABELS = {
    "level": "Nivel (sin transformar)",
    "diff": "Diferencia anual (nivel)",
    "yoy": "Variación interanual (%)",
    "logdiff": "Log-diferencia anual (%)",
}


# ---------------------------------------------------------------------------
# Selección de transformaciones (ADF/KPSS) — se corre una vez y se cachea
# ---------------------------------------------------------------------------
def select_transforms_for_all_indicators(df: pd.DataFrame) -> dict:
    """
    Corre select_best_transform() para cada indicador de las 3 categorías.

    Returns
    -------
    dict {code: {"chosen_transform": str, "results": {...}, "indicator_type": str}}
    """
    selections = {}
    for category, indicators in CATEGORY_DEFINITIONS.items():
        for code, cfg in indicators.items():
            if code not in df.columns:
                continue
            outcome = select_best_transform(df[code], cfg["indicator_type"])
            outcome["indicator_type"] = cfg["indicator_type"]
            selections[code] = outcome
    return selections


def transformed_indicator_table(df: pd.DataFrame, transform_selections: dict) -> pd.DataFrame:
    """
    Construye una tabla con la serie YA TRANSFORMADA (según la transform
    elegida por ADF/KPSS) para cada indicador de las 3 categorías, en una
    sola tabla — esta es la tabla histórica "ordenada por mes" que pide la
    pestaña 1.
    """
    columns = {}
    for category, indicators in CATEGORY_DEFINITIONS.items():
        for code, cfg in indicators.items():
            if code not in df.columns or code not in transform_selections:
                continue
            transform = transform_selections[code]["chosen_transform"]
            columns[code] = apply_candidate_transform(df[code], transform)
    return pd.concat(columns, axis=1)


# ---------------------------------------------------------------------------
# Z-Score individual
# ---------------------------------------------------------------------------
def rolling_or_full_zscore(series: pd.Series, window_years: int | None = 10) -> pd.Series:
    """
    window_years=None -> normaliza con media/desviación de TODA la muestra.
    window_years=N    -> normaliza con ventana móvil de N años (recomendado
                          por el Chicago Fed Letter de 2008 para evitar el
                          "moving target problem").
    """
    s = series.dropna()
    if window_years is None:
        mean, std = s.mean(), s.std()
    else:
        window_days = int(window_years * 365)
        mean = s.rolling(window_days, min_periods=window_days // 2).mean()
        std = s.rolling(window_days, min_periods=window_days // 2).std()
    z = (s - mean) / std
    z.name = f"{series.name}_z"
    return z


def _normalize_weights(weights: dict[str, float]) -> pd.Series:
    """Normaliza los pesos para que sumen 1 (avisa al llamador si no sumaban 1)."""
    w = pd.Series(weights, dtype=float)
    total = w.sum()
    if total <= 0:
        raise ValueError("La suma de los pesos debe ser positiva.")
    return w / total


# ---------------------------------------------------------------------------
# Z-Score de categoría (Nivel 2, con doble estandarización)
# ---------------------------------------------------------------------------
def category_zscore(
    df: pd.DataFrame,
    category: str,
    transform_selections: dict,
    weights: dict[str, float] | None = None,
    window_years: int | None = 10,
) -> dict:
    """
    Calcula el Z-Score de una categoría con doble estandarización.

    Returns
    -------
    dict con:
      - "individual_z": DataFrame de Z-Scores individuales (uno por indicador)
      - "weights_normalized": pd.Series de pesos ya normalizados (suman 1)
      - "raw_composite": pd.Series, promedio ponderado de Z-Scores (antes de re-estandarizar)
      - "category_zscore": pd.Series, EL Z-SCORE FINAL de la categoría (re-estandarizado)
      - "contributions": DataFrame, aporte de cada indicador al category_zscore en cada fecha
      - "baseline": float, término constante de ajuste de media (ver decomposición)
    """
    config = CATEGORY_DEFINITIONS[category]
    if weights is None:
        weights = {code: 1.0 for code in config}
    w = _normalize_weights(weights)

    z_components = {}
    for code, cfg in config.items():
        if code not in df.columns or code not in transform_selections:
            continue
        transform = transform_selections[code]["chosen_transform"]
        raw = apply_candidate_transform(df[code], transform)
        z = rolling_or_full_zscore(raw, window_years=window_years)
        z_components[code] = z * cfg["sign"]

    z_df = pd.concat(z_components, axis=1)
    w_aligned = w.reindex(z_df.columns).fillna(0.0)

    # Nivel 2a: promedio ponderado de Z-Scores individuales (ignorando NaN puntuales)
    weighted = z_df.mul(w_aligned, axis=1)
    available_weight = z_df.notna().mul(w_aligned, axis=1).sum(axis=1)
    raw_composite = weighted.sum(axis=1) / available_weight.replace(0, np.nan)

    # Nivel 2b: RE-ESTANDARIZAR el compuesto ponderado (la "doble estandarización")
    mean_c, std_c = raw_composite.mean(), raw_composite.std()
    category_z = (raw_composite - mean_c) / std_c
    category_z.name = f"{category}_zscore"

    # Descomposición: contribution_i_t = w_i * z_i_t / std_c, de forma que
    # sum_i(contribution_i_t) + baseline == category_z_t exactamente.
    contributions = weighted.div(std_c)
    baseline = -mean_c / std_c

    return {
        "individual_z": z_df,
        "weights_normalized": w_aligned,
        "raw_composite": raw_composite,
        "category_zscore": category_z,
        "contributions": contributions,
        "baseline": baseline,
    }


# ---------------------------------------------------------------------------
# Z-Score maestro (Nivel 3, también con doble estandarización)
# ---------------------------------------------------------------------------
def master_zscore(
    category_results: dict[str, dict],
    weights: dict[str, float] | None = None,
) -> dict:
    """
    Combina los 3 Z-Scores de categoría (ya re-estandarizados) en un único
    Z-Score maestro, también re-estandarizado, con pesos libres entre
    categorías.

    Parameters
    ----------
    category_results : dict {"growth": resultado de category_zscore(...), ...}
    weights : pesos entre categorías (p. ej. {"growth": 0.2, "inflation": 0.4, "employment": 0.4})
    """
    if weights is None:
        weights = {k: 1.0 for k in category_results}
    w = _normalize_weights(weights)

    cat_z = pd.concat(
        {cat: res["category_zscore"] for cat, res in category_results.items()}, axis=1
    )
    w_aligned = w.reindex(cat_z.columns).fillna(0.0)

    weighted = cat_z.mul(w_aligned, axis=1)
    available_weight = cat_z.notna().mul(w_aligned, axis=1).sum(axis=1)
    raw_composite = weighted.sum(axis=1) / available_weight.replace(0, np.nan)

    mean_c, std_c = raw_composite.mean(), raw_composite.std()
    master_z = (raw_composite - mean_c) / std_c
    master_z.name = "master_zscore"

    contributions = weighted.div(std_c)
    baseline = -mean_c / std_c

    return {
        "weights_normalized": w_aligned,
        "raw_composite": raw_composite,
        "master_zscore": master_z,
        "contributions": contributions,
        "baseline": baseline,
    }


def decompose_at_date(contributions: pd.DataFrame, baseline: float, date) -> pd.Series:
    """
    Extrae la descomposición (aportes de cada componente) para una fecha
    específica, lista para graficar como barras. Incluye el baseline como
    una entrada más, de forma que la suma de todas las barras reproduce
    exactamente el Z-Score final de esa fecha.
    """
    row = contributions.loc[date].copy()
    row["(ajuste de media histórica)"] = baseline
    return row.sort_values()
