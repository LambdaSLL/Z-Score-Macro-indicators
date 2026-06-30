"""
stationarity.py
----------------
Selecciona la transformación adecuada para cada indicador macro de forma
empírica, en vez de aplicar una regla de dedo única (p. ej. "todo en YoY").
Sigue el espíritu de McCracken & Ng (2016, FRED-MD): cada serie recibe su
propio código de transformación según pruebas de raíz unitaria, no un
criterio uniforme para todas.

Se combinan dos pruebas complementarias porque tienen hipótesis nulas
opuestas, y usarlas juntas reduce el riesgo de una conclusión errónea por
baja potencia de una sola prueba:

  - ADF (Augmented Dickey-Fuller): H0 = la serie tiene raíz unitaria
    (no estacionaria). p < 0.05 → se rechaza H0 → evidencia de
    estacionariedad.
  - KPSS (Kwiatkowski-Phillips-Schmidt-Shin): H0 = la serie ES
    estacionaria. p >= 0.05 → no se rechaza H0 → evidencia de
    estacionariedad.

Veredicto:
  - "stationary"      ambas pruebas coinciden en que es estacionaria
  - "non_stationary"   ambas coinciden en que NO es estacionaria
  - "inconclusive"     las pruebas discrepan (ocurre con cierta frecuencia
                        en series macro reales; ver Ng & Perron)
"""

from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss


def _safe_adf_pvalue(series: pd.Series) -> float | None:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, p_value, *_ = adfuller(series, autolag="AIC")
        return float(p_value)
    except Exception:
        return None


def _safe_kpss_pvalue(series: pd.Series) -> float | None:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, p_value, *_ = kpss(series, regression="c", nlags="auto")
        return float(p_value)
    except Exception:
        return None


def test_stationarity(series: pd.Series, min_obs: int = 30) -> dict:
    """Corre ADF y KPSS sobre una serie y devuelve p-valores + veredicto."""
    s = series.dropna()
    if len(s) < min_obs:
        return {"adf_pvalue": None, "kpss_pvalue": None, "verdict": "insufficient_data", "n_obs": len(s)}

    adf_p = _safe_adf_pvalue(s)
    kpss_p = _safe_kpss_pvalue(s)

    if adf_p is None or kpss_p is None:
        verdict = "insufficient_data"
    else:
        adf_says_stationary = adf_p < 0.05
        kpss_says_stationary = kpss_p >= 0.05
        if adf_says_stationary and kpss_says_stationary:
            verdict = "stationary"
        elif (not adf_says_stationary) and (not kpss_says_stationary):
            verdict = "non_stationary"
        else:
            verdict = "inconclusive"

    return {"adf_pvalue": adf_p, "kpss_pvalue": kpss_p, "verdict": verdict, "n_obs": len(s)}


# ---------------------------------------------------------------------------
# Candidatos de transformación por tipo de indicador
# ---------------------------------------------------------------------------
# "rate"  -> series ya expresadas en % o puntos (tasas, breakevens, utilización):
#            tiene sentido probar nivel y primera diferencia, NO YoY/log
#            (no tiene sentido sacar "variación interanual de una tasa de
#            desempleo en %" como si fuera un índice).
# "index" -> series en niveles con tendencia (PIB, producción, nóminas,
#            salarios, precios): tiene sentido probar nivel, diferencia,
#            YoY y log-diferencia, siguiendo los códigos de McCracken & Ng.

RATE_CANDIDATES = ["level", "diff"]
INDEX_CANDIDATES = ["level", "diff", "yoy", "logdiff"]

# Orden de preferencia: ante empate o resultado "inconclusive", se prefiere
# la transformación MENOS agresiva (más fácil de interpretar, menos ruido).
PREFERENCE_ORDER = ["level", "yoy", "diff", "logdiff"]


def apply_candidate_transform(series: pd.Series, transform: str) -> pd.Series:
    s = series.dropna()
    if transform == "level":
        return s
    if transform == "diff":
        return s.diff(12)  # 12 meses = 1 año (datos ya a frecuencia mensual)
    if transform == "yoy":
        return 100 * s.pct_change(12)
    if transform == "logdiff":
        return 100 * np.log(s).diff(12)
    raise ValueError(f"Transformación no reconocida: {transform}")


def select_best_transform(series: pd.Series, indicator_type: str) -> dict:
    """
    Prueba cada transformación candidata (según el tipo de indicador) con
    ADF + KPSS y selecciona la transformación menos agresiva que resulte
    estadísticamente estacionaria.

    Returns
    -------
    dict con:
      - "chosen_transform": la transformación seleccionada
      - "results": {transform: {adf_pvalue, kpss_pvalue, verdict, n_obs}}
        para las 2-4 transformaciones probadas, de forma transparente
    """
    candidates = RATE_CANDIDATES if indicator_type == "rate" else INDEX_CANDIDATES

    results = {}
    for transform in candidates:
        try:
            transformed = apply_candidate_transform(series, transform)
            results[transform] = test_stationarity(transformed)
        except Exception:
            results[transform] = {"adf_pvalue": None, "kpss_pvalue": None, "verdict": "error", "n_obs": 0}

    # 1) Preferir la transformación menos agresiva que sea "stationary" por ambas pruebas.
    for transform in PREFERENCE_ORDER:
        if transform in results and results[transform]["verdict"] == "stationary":
            return {"chosen_transform": transform, "results": results}

    # 2) Si ninguna es concluyente, aceptar la menos agresiva "inconclusive".
    for transform in PREFERENCE_ORDER:
        if transform in results and results[transform]["verdict"] == "inconclusive":
            return {"chosen_transform": transform, "results": results}

    # 3) Último recurso: la transformación más agresiva disponible (suele
    #    eliminar la raíz unitaria incluso si las pruebas no son concluyentes).
    fallback = candidates[-1]
    return {"chosen_transform": fallback, "results": results}
