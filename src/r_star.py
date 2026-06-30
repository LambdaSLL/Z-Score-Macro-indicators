"""
r_star.py
---------
Métodos de REFERENCIA para estimar r* (la tasa de interés real neutral,
r_n en la notación del paper de Marín & Chacón). Incluye:

  1) Proxies calculables 100% con datos de FRED (TIPS, crecimiento de
     tendencia).
  2) Descarga en vivo del modelo oficial Holston-Laubach-Williams (HLW) /
     Laubach-Williams (LW) directamente del sitio del NY Fed, con manejo
     de fallos y valor de respaldo — siguiendo el mismo patrón robusto de
     fetch_rstar() descrito por el usuario para su otra app de Fair Value.

Ninguno reemplaza el juicio del PM; todos son puntos de anclaje.
"""

from __future__ import annotations
import pandas as pd
import numpy as np


def r_star_market_tips(df: pd.DataFrame) -> pd.Series:
    """
    Método 1: usar el rendimiento real de mercado de los TIPS a 10 años
    (DFII10) como proxy directo de r* "implícito en el mercado".

    Ventaja: se actualiza diariamente y refleja expectativas de mercado en
    tiempo real. Desventaja: incluye primas de plazo y de liquidez, no es
    una r* "estructural" en el sentido de Laubach-Williams.
    """
    s = df["DFII10"].dropna()
    s.name = "r_star_tips10"
    return s


def r_star_long_run_growth(df: pd.DataFrame, window_years: int = 10) -> pd.Series:
    """
    Método 2: heurística de practicante muy usada — en el largo plazo,
    bajo una regla de oro simplificada, r* tiende a aproximarse a la tasa
    de crecimiento real de tendencia de la economía. Se calcula como el
    promedio móvil de `window_years` años del crecimiento real del PIB.

    Ventaja: fácil de calcular y de explicar a un comité de inversión.
    Desventaja: ignora ahorro, demografía y demanda de activos seguros,
    factores que la literatura (Laubach-Williams, Holston et al.) muestra
    que han bajado r* estructuralmente desde los 1980s.
    """
    real = df["GDPC1"].dropna()
    yoy_growth = 100 * real.pct_change(4)
    r_star = yoy_growth.rolling(window=window_years * 4, min_periods=8).mean()
    r_star.name = "r_star_long_run_growth"
    return r_star


# ---------------------------------------------------------------------------
# Método 3: descarga en vivo del modelo oficial HLW/LW del NY Fed
# ---------------------------------------------------------------------------
# URLs verificadas directamente en https://www.newyorkfed.org/research/policy/rstar
# (junio 2026). El NY Fed las ha cambiado de nombre/ubicación en el pasado
# (p. ej. al introducir el modelo ajustado por COVID), por lo que se intenta
# más de una URL y se cae a un valor fijo si todas fallan — mismo patrón que
# usaste en tu otra app de Fair Value.

NYFED_HLW_URL = (
    "https://www.newyorkfed.org/medialibrary/media/research/economists/"
    "williams/data/Holston_Laubach_Williams_current_estimates.xlsx"
)
NYFED_LW_URL = (
    "https://www.newyorkfed.org/medialibrary/media/research/economists/"
    "williams/data/Laubach_Williams_current_estimates.xlsx"
)
# URLs "legacy" conocidas, por si el NY Fed reestructura el sitio de nuevo
# (precedente documentado: el archivo vivió antes en frbsf.org y en otra
# ruta de newyorkfed.org bajo /research/policy/rstar/).
NYFED_FALLBACK_URLS = [
    "https://www.newyorkfed.org/medialibrary/media/research/policy/rstar/"
    "Holston_Laubach_Williams_current_estimates.xlsx",
    "https://www.frbsf.org/wp-content/uploads/Laubach_Williams_updated_estimates.xlsx",
]

DEFAULT_RSTAR_FALLBACK = 1.46  # valor de respaldo fijo (referencia Credicorp)


def _find_rstar_series_in_workbook(url: str, keyword: str = "rstar", timeout: int = 15) -> pd.Series | None:
    """
    Intenta localizar y parsear la columna r* dentro de un archivo .xlsx del
    NY Fed, sin asumir una estructura de fila/columna fija (el archivo ha
    cambiado de formato en el pasado). Recorre todas las hojas, busca la
    fila de encabezado que contiene la palabra clave "rstar" y extrae la
    columna de fecha y la de valores.

    Devuelve None si no logra encontrar nada parseable (el llamador decide
    si intenta la siguiente URL o usa el valor de respaldo).
    """
    try:
        sheets = pd.read_excel(url, sheet_name=None, header=None, engine="openpyxl")
    except Exception:
        return None

    for _, sheet_df in sheets.items():
        max_scan_rows = min(20, len(sheet_df))
        for row_idx in range(max_scan_rows):
            row_as_text = sheet_df.iloc[row_idx].astype(str).str.lower()
            if not row_as_text.str.contains(keyword).any():
                continue

            header_row = sheet_df.iloc[row_idx]
            data = sheet_df.iloc[row_idx + 1:].copy()
            data.columns = header_row

            date_col = next(
                (c for c in data.columns if isinstance(c, str) and ("date" in c.lower() or "quarter" in c.lower())),
                data.columns[0],
            )
            rstar_col = next(
                (c for c in data.columns if isinstance(c, str) and keyword in c.lower().replace(" ", "").replace("*", "")),
                None,
            )
            if rstar_col is None:
                continue

            dates = pd.to_datetime(data[date_col], errors="coerce")
            values = pd.to_numeric(data[rstar_col], errors="coerce")
            series = pd.Series(values.values, index=dates).dropna()
            if len(series) > 0:
                series.name = "r_star_nyfed"
                return series.sort_index()

    return None


def fetch_rstar_nyfed(
    manual_override: float | None = None,
    fallback_value: float = DEFAULT_RSTAR_FALLBACK,
) -> dict:
    """
    Réplica del patrón fetch_rstar() descrito por el usuario para su app de
    Fair Value: intenta descargar el r* oficial (HLW, luego LW) en vivo del
    NY Fed; si todo falla, usa un valor de respaldo fijo. Permite además un
    override manual que tiene prioridad absoluta (para forzarlo desde la
    barra lateral de Streamlit).

    Returns
    -------
    dict con:
      - "value": float, el valor de r* a usar (manual > vivo > respaldo)
      - "series": pd.Series trimestral si la descarga fue exitosa, si no None
      - "source": "manual" | "live" | "fallback"
      - "url": la URL que funcionó (si source == "live"), si no None
    """
    if manual_override is not None:
        return {"value": manual_override, "series": None, "source": "manual", "url": None}

    urls_to_try = [NYFED_HLW_URL, NYFED_LW_URL] + NYFED_FALLBACK_URLS
    for url in urls_to_try:
        series = _find_rstar_series_in_workbook(url)
        if series is not None:
            return {"value": float(series.iloc[-1]), "series": series, "source": "live", "url": url}

    return {"value": fallback_value, "series": None, "source": "fallback", "url": None}


def r_star_fixed_reference(value: float = DEFAULT_RSTAR_FALLBACK) -> float:
    """
    Mantiene compatibilidad con el código anterior: devuelve simplemente el
    valor de respaldo fijo, sin intentar la descarga en vivo. Para la
    versión con descarga + fallback automático, usar fetch_rstar_nyfed().
    """
    return value


def all_r_star_estimates(df: pd.DataFrame) -> pd.DataFrame:
    """Combina los métodos automatizables vía FRED en una sola tabla."""
    estimates = [
        r_star_market_tips(df),
        r_star_long_run_growth(df),
    ]
    combined = pd.concat(estimates, axis=1)
    return combined
