"""
output_gap.py
--------------
Métodos de REFERENCIA para estimar la brecha de producto (output gap).
Ninguno se impone como "el correcto": el PM elige cuál usar, o promedia
varios, en el dashboard. Todos devuelven un porcentaje (PIB real vs.
potencial, o equivalente vía Ley de Okun).
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from statsmodels.tsa.filters.hp_filter import hpfilter


def output_gap_cbo(df: pd.DataFrame) -> pd.Series:
    """
    Método 1 (más simple y "oficial"): usa directamente el PIB potencial
    estimado por la CBO (serie GDPPOT de FRED) contra el PIB real (GDPC1).

        gap_t = 100 * (GDPC1_t - GDPPOT_t) / GDPPOT_t

    Ventaja: es el mismo insumo que usa la Fed y el CBO. Desventaja: el
    PIB potencial del CBO se revisa con rezago y de forma poco frecuente.
    """
    real = df["GDPC1"].dropna()
    potential = df["GDPPOT"].dropna()
    joined = pd.concat([real, potential], axis=1).dropna()
    gap = 100 * (joined["GDPC1"] - joined["GDPPOT"]) / joined["GDPPOT"]
    gap.name = "output_gap_cbo"
    return gap


def output_gap_hp_filter(df: pd.DataFrame, lamb: float = 1600) -> pd.Series:
    """
    Método 2: filtro de Hodrick-Prescott aplicado al log del PIB real para
    extraer la tendencia (proxy del producto potencial) y obtener el gap
    como el componente cíclico.

        log(GDPC1) = tendencia_HP + ciclo_HP
        gap_t = 100 * ciclo_HP_t

    lamb=1600 es el valor estándar para datos trimestrales (Hodrick-Prescott,
    1997). Ventaja: no depende de estimaciones externas (CBO). Desventaja:
    sensible al final de la muestra ("end-point problem") y a lamb elegido.
    """
    log_gdp = np.log(df["GDPC1"].dropna())
    cycle, trend = hpfilter(log_gdp, lamb=lamb)
    gap = 100 * cycle
    gap.name = "output_gap_hp"
    return gap


def output_gap_okun(df: pd.DataFrame, okun_coefficient: float = 2.0) -> pd.Series:
    """
    Método 3: Ley de Okun, usando la brecha de desempleo (tasa de
    desempleo natural NROU del CBO menos la tasa observada UNRATE) como
    proxy del output gap.

        gap_t = okun_coefficient * (NROU_t - UNRATE_t)

    okun_coefficient=2.0 es la regla de dedo clásica (~2 puntos de PIB por
    cada punto de desempleo por debajo del natural). Ventaja: usa datos
    mensuales de desempleo, más frecuentes que el PIB. Desventaja: el
    coeficiente de Okun varía en el tiempo y entre episodios.
    """
    nrou = df["NROU"].dropna()
    unrate = df["UNRATE"].dropna()
    joined = pd.concat([nrou, unrate], axis=1).dropna()
    gap = okun_coefficient * (joined["NROU"] - joined["UNRATE"])
    gap.name = "output_gap_okun"
    return gap


def output_growth_gap_paper(df: pd.DataFrame, lookback_years: int = 10) -> pd.Series:
    """
    Método 4 (el usado en el paper de Marín & Chacón, Anexo A1): en vez de
    una brecha de NIVEL, usa una brecha de TASA DE CRECIMIENTO esperado del
    PIB real respecto a su tendencia de largo plazo (promedio móvil de
    `lookback_years` años de crecimiento real).

        gap_t = PIBe_t - PIBtendencia_t

    donde PIBe es el crecimiento real interanual observado/esperado y
    PIBtendencia es el promedio móvil de largo plazo. Esta es la
    especificación exacta de la ecuación (A1) del paper.
    """
    real = df["GDPC1"].dropna()
    yoy_growth = 100 * real.pct_change(4)  # crecimiento interanual (datos trimestrales)
    trend_growth = yoy_growth.rolling(window=lookback_years * 4, min_periods=8).mean()
    gap = yoy_growth - trend_growth
    gap.name = "output_growth_gap_paper"
    return gap


def all_output_gap_estimates(df: pd.DataFrame) -> pd.DataFrame:
    """Combina los 4 métodos en una sola tabla para comparar en el dashboard."""
    estimates = [
        output_gap_cbo(df),
        output_gap_hp_filter(df),
        output_gap_okun(df),
        output_growth_gap_paper(df),
    ]
    combined = pd.concat(estimates, axis=1)
    return combined
