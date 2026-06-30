"""
zscore_indicators.py
---------------------
Construye un indicador Z-Score compuesto de desviación macro, siguiendo la
misma lógica que el Chicago Fed National Activity Index (CFNAI) y el
Aruoba-Diebold-Scotti (ADS) Index, pero simplificado en 3 sub-índices
(crecimiento, inflación, empleo) con PESOS LIBRES definidos por el usuario
(PM), en vez de pesos fijados por PCA.

Referencias metodológicas:
- Stock, J. H., & Watson, M. W. (1999). Forecasting inflation. Journal of
  Monetary Economics, 44(2), 293-335. (origen del enfoque de "factor común"
  vía estandarización + ponderación que usa el CFNAI)
- Federal Reserve Bank of Chicago. "Background on the CFNAI": cada serie se
  transforma, se de-media y se estandariza (z-score) antes de ponderar.
- Aruoba, S. B., Diebold, F. X., & Scotti, C. (2009). Real-Time Measurement
  of Business Conditions. Journal of Business and Economic Statistics,
  27(4), 417-427.

Diferencia deliberada con el CFNAI: aquí los pesos NO se estiman vía PCA,
sino que quedan 100% en manos del usuario, en línea con la filosofía de
calibración libre usada en taylor_rule.py.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Definición de indicadores por categoría
# ---------------------------------------------------------------------------
# transform: "level" (usar el nivel tal cual, ya en %), "yoy" (variación
#            interanual %), "mom_diff" (variación mensual en nivel),
#            "mom_pct" (variación mensual %)
# sign: 1 si "más alto = mejor" para esa categoría, -1 si "más alto = peor"
#       (p. ej. desempleo o solicitudes de desempleo deben invertirse)

GROWTH_INDICATORS = {
    "GDPC1":   {"transform": "yoy",      "sign": 1, "label": "PIB real (YoY %)"},
    "INDPRO":  {"transform": "yoy",      "sign": 1, "label": "Producción industrial (YoY %)"},
    "RSAFS":   {"transform": "yoy",      "sign": 1, "label": "Ventas minoristas (YoY %)"},
    "CUMFNS":  {"transform": "level",    "sign": 1, "label": "Utilización de capacidad (%)"},
}

INFLATION_INDICATORS = {
    "CPIAUCSL_YOY": {"transform": "level", "sign": 1, "label": "CPI total (YoY %)"},
    "PCEPI_YOY":    {"transform": "level", "sign": 1, "label": "PCE (YoY %)"},
    "T5YIE":        {"transform": "level", "sign": 1, "label": "Breakeven inflación 5y (%)"},
    "T10YIE":       {"transform": "level", "sign": 1, "label": "Breakeven inflación 10y (%)"},
    "MICH":         {"transform": "level", "sign": 1, "label": "Expectativas U. Michigan 1y (%)"},
}

EMPLOYMENT_INDICATORS = {
    "PAYEMS":  {"transform": "mom_diff", "sign": 1,  "label": "Cambio mensual nóminas (miles)"},
    "UNRATE":  {"transform": "level",    "sign": -1, "label": "Tasa de desempleo (%, invertida)"},
    "ICSA":    {"transform": "level",    "sign": -1, "label": "Solicitudes desempleo semanal (invertida)"},
    "AHETPI":  {"transform": "yoy",      "sign": 1,  "label": "Salario promedio por hora (YoY %, invertido por inflación opcional)"},
}

CATEGORY_DEFINITIONS = {
    "growth": GROWTH_INDICATORS,
    "inflation": INFLATION_INDICATORS,
    "employment": EMPLOYMENT_INDICATORS,
}


def apply_transform(series: pd.Series, transform: str) -> pd.Series:
    """Aplica la transformación definida para un indicador dado."""
    s = series.dropna()
    if transform == "level":
        return s
    if transform == "yoy":
        return 100 * s.pct_change(365)  # datos ya llevados a frecuencia diaria
    if transform == "mom_diff":
        return s.diff(30)
    if transform == "mom_pct":
        return 100 * s.pct_change(30)
    raise ValueError(f"Transformación no reconocida: {transform}")


def rolling_or_full_zscore(series: pd.Series, window_years: int | None = 10) -> pd.Series:
    """
    Calcula el Z-Score de una serie.

    window_years=None  -> normaliza con la media/desviación de TODA la
                            muestra disponible (estilo CFNAI estándar).
    window_years=N      -> normaliza con una ventana móvil de N años, para
                            permitir que la "normalidad" se ajuste con el
                            tiempo (recomendado por el Chicago Fed Letter de
                            2008 para evitar el "moving target problem").
    """
    s = series.dropna()
    if window_years is None:
        mean = s.mean()
        std = s.std()
        z = (s - mean) / std
    else:
        window_days = int(window_years * 365)
        mean = s.rolling(window_days, min_periods=window_days // 2).mean()
        std = s.rolling(window_days, min_periods=window_days // 2).std()
        z = (s - mean) / std
    z.name = f"{series.name}_z"
    return z


def _weighted_avg_ignore_nan(z_df: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    """
    Promedio ponderado fila a fila, ignorando NaN y renormalizando los
    pesos disponibles en cada fecha (para que un dato faltante puntual no
    tumbe todo el índice a NaN).
    """
    w = pd.Series(weights).reindex(z_df.columns).fillna(0.0)
    weighted = z_df.mul(w, axis=1)
    available_weight = z_df.notna().mul(w, axis=1).sum(axis=1)
    combined = weighted.sum(axis=1) / available_weight.replace(0, np.nan)
    return combined


def category_zscore(
    df: pd.DataFrame,
    category: str,
    weights: dict[str, float] | None = None,
    window_years: int | None = 10,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Calcula el Z-Score compuesto de una categoría (growth/inflation/employment).

    Returns
    -------
    (serie_z_compuesta, dataframe_con_z_individuales)
    """
    config = CATEGORY_DEFINITIONS[category]
    if weights is None:
        weights = {code: 1.0 for code in config}

    z_components = {}
    for code, cfg in config.items():
        if code not in df.columns:
            continue
        raw = apply_transform(df[code], cfg["transform"])
        z = rolling_or_full_zscore(raw, window_years=window_years)
        z_components[code] = z * cfg["sign"]

    z_df = pd.concat(z_components, axis=1)
    composite = _weighted_avg_ignore_nan(z_df, weights)
    composite.name = f"{category}_zscore"
    return composite, z_df


def master_zscore(
    growth_z: pd.Series,
    inflation_z: pd.Series,
    employment_z: pd.Series,
    weights: dict[str, float] | None = None,
) -> pd.Series:
    """
    Combina los 3 sub-índices en un único Z-Score maestro de desviación
    macro, con pesos libres entre categorías.
    """
    if weights is None:
        weights = {"growth": 1.0, "inflation": 1.0, "employment": 1.0}

    combined_df = pd.concat(
        {"growth": growth_z, "inflation": inflation_z, "employment": employment_z}, axis=1
    )
    master = _weighted_avg_ignore_nan(combined_df, weights)
    master.name = "master_zscore"
    return master


def interpret_zscore(value: float) -> str:
    """Traduce el valor del Z-Score maestro en una lectura cualitativa simple."""
    if value >= 1.5:
        return "Muy por encima del promedio histórico (fuerte expansión / sobrecalentamiento)"
    elif value >= 0.5:
        return "Por encima del promedio histórico (expansión moderada)"
    elif value > -0.5:
        return "Cercano al promedio histórico (crecimiento en línea con tendencia)"
    elif value > -1.5:
        return "Por debajo del promedio histórico (debilidad moderada)"
    else:
        return "Muy por debajo del promedio histórico (contracción / señal de alerta)"
