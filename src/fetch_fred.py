"""
fetch_fred.py
--------------
Descarga y cachea localmente las series de FRED necesarias para:
  1) Calcular la Regla de Taylor (varias variantes de r*, inflación, output gap)
  2) Monitorear la curva del Tesoro (2y, 10y, spread 2s10s, etc.)

Requiere una FRED_API_KEY gratuita: https://fredaccount.stlouisfed.org/apikeys

Todas las series usadas son públicas y gratuitas. No se usa ningún dato de
Bloomberg ni de proveedores propietarios (a diferencia del paper original).
"""

from __future__ import annotations
import os
import pandas as pd
from pathlib import Path
from fredapi import Fred
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)
CACHE_FILE = DATA_DIR / "fred_series.parquet"

# Mapa: nombre interno -> código de serie en FRED
SERIES_MAP = {
    # Curva del Tesoro
    "DGS3MO": "DGS3MO",       # 3 meses
    "DGS2": "DGS2",           # 2 años
    "DGS10": "DGS10",         # 10 años
    "DGS30": "DGS30",         # 30 años
    "T10Y2Y": "T10Y2Y",       # spread 10y-2y (ya calculado por FRED)
    "T10Y3M": "T10Y3M",       # spread 10y-3m

    # Política monetaria
    "FEDFUNDS": "FEDFUNDS",   # tasa de fondos federales efectiva (mensual)
    "DFEDTARU": "DFEDTARU",   # límite superior del rango objetivo (diario)

    # Inflación
    "CPIAUCSL": "CPIAUCSL",   # CPI total, índice, mensual
    "CPILFESL": "CPILFESL",   # CPI core
    "PCEPI": "PCEPI",         # deflactor PCE (medida preferida de la Fed)
    "PCEPILFE": "PCEPILFE",   # PCE core
    "T5YIE": "T5YIE",         # breakeven de inflación a 5 años
    "T10YIE": "T10YIE",       # breakeven de inflación a 10 años
    "MICH": "MICH",           # expectativas de inflación U. Michigan (1 año)

    # Actividad real / output gap
    "GDPC1": "GDPC1",         # PIB real, trimestral
    "GDPPOT": "GDPPOT",       # PIB potencial (estimado por CBO), trimestral
    "UNRATE": "UNRATE",       # tasa de desempleo, mensual
    "NROU": "NROU",           # tasa natural de desempleo (CBO), trimestral
    "CUMFNS": "CUMFNS",        # tasa de utilización de capacidad (sustituto de CPMFTOT de Bloomberg)
    "INDPRO": "INDPRO",       # producción industrial, mensual
    "RSAFS": "RSAFS",         # ventas minoristas, mensual

    # Empleo (para el indicador Z-Score de empleo)
    "PAYEMS": "PAYEMS",       # nóminas no agrícolas, mensual (nivel, miles de empleos)
    "ICSA": "ICSA",           # solicitudes iniciales de desempleo, semanal
    "AHETPI": "CES0500000003",  # salario promedio por hora, mensual

    # Mercado / curva real
    "DFII10": "DFII10",       # TIPS 10 años (proxy de tasa real de mercado / r* de mercado)
}


def get_fred_client() -> Fred:
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise RuntimeError(
            "No se encontró FRED_API_KEY. Crea un archivo .env a partir de "
            ".env.example con tu API key gratuita de FRED."
        )
    return Fred(api_key=api_key)


def fetch_all_series(force_refresh: bool = False) -> pd.DataFrame:
    """
    Descarga todas las series definidas en SERIES_MAP y las combina en un
    único DataFrame con índice de fecha diario (forward-filled para series
    de menor frecuencia, p. ej. mensuales o trimestrales).

    Si force_refresh=False y existe un cache local, lo reutiliza.
    """
    if CACHE_FILE.exists() and not force_refresh:
        return pd.read_parquet(CACHE_FILE)

    fred = get_fred_client()
    series_dict = {}
    for internal_name, fred_code in SERIES_MAP.items():
        try:
            s = fred.get_series(fred_code)
            s.name = internal_name
            series_dict[internal_name] = s
        except Exception as e:
            print(f"[WARN] No se pudo descargar {fred_code} ({internal_name}): {e}")

    df = pd.concat(series_dict.values(), axis=1)
    df = df.sort_index()

    # Algunas series (p. ej. GDPPOT, el PIB potencial del CBO) incluyen
    # PROYECCIONES A FUTURO, no solo histórico. Si se usa el rango completo
    # del índice combinado, la tabla diaria se extendería años hacia
    # adelante, y el resto de series (que sí terminan en el presente)
    # quedarían "congeladas" en su último valor real durante todo ese
    # tramo futuro — distorsionando cualquier cálculo que las combine (p.
    # ej. el output gap, al comparar un PIB real congelado contra un PIB
    # potencial que sigue creciendo en la proyección). Por eso se trunca
    # explícitamente el índice a HOY antes de propagar valores.
    today = pd.Timestamp.now().normalize()
    end_date = min(df.index.max(), today)
    full_range = pd.date_range(df.index.min(), end_date, freq="D")

    # Llevar todo a frecuencia diaria, propagando el último valor disponible
    df = df.reindex(full_range).ffill()

    # CPIAUCSL, CPILFESL, PCEPI y PCEPILFE vienen como NIVELES de índice, no
    # como tasas de inflación. Se agregan columnas derivadas con la
    # variación interanual (%) para poder usarlas directamente como
    # "inflación esperada/observada" en la Regla de Taylor.
    for level_col in ["CPIAUCSL", "CPILFESL", "PCEPI", "PCEPILFE"]:
        if level_col in df.columns:
            # 365 pasos diarios ~ 1 año, dado que ya se llevó todo a diario
            df[f"{level_col}_YOY"] = 100 * df[level_col].pct_change(365)

    df.to_parquet(CACHE_FILE)
    return df


def latest_value(df: pd.DataFrame, column: str) -> float | None:
    """Devuelve el último valor no nulo disponible de una columna."""
    series = df[column].dropna()
    if series.empty:
        return None
    return float(series.iloc[-1])


if __name__ == "__main__":
    data = fetch_all_series(force_refresh=True)
    print(data.tail())
    print(f"\nGuardado en: {CACHE_FILE}")
