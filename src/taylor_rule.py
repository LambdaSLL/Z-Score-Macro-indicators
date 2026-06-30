"""
taylor_rule.py
--------------
Implementación genérica de la Regla de Taylor, siguiendo la notación del
Anexo A1 del paper de Marín & Chacón (2026):

    r_o = r_n + i_e + 0.5*(PIBe - PIBt) + 0.5*(i_e - i_o)         (A1)

Generalizada con pesos libres (no fijos a 0.5/0.5) para que el PM pueda
calibrar la sensibilidad de la regla:

    r_o = r_n + i_e + w_y*(output_gap) + w_pi*(i_e - i_o)

Todos los parámetros (r_n, i_o, w_y, w_pi) quedan como argumentos
explícitos — el PM los fija en el dashboard, usando las tablas de
"métodos de referencia" de output_gap.py y r_star.py solo como guía,
nunca como un valor impuesto por el código.
"""

from __future__ import annotations
from dataclasses import dataclass
import pandas as pd


@dataclass
class TaylorRuleParams:
    """Parámetros de calibración libre del PM."""
    r_star: float = 0.5          # tasa real neutral (r_n), en %
    inflation_target: float = 2.0  # meta de inflación (i_o), en %
    weight_output_gap: float = 0.5  # peso w_y sobre la brecha de producto
    weight_inflation_gap: float = 0.5  # peso w_pi sobre la brecha de inflación


def taylor_rate(
    expected_inflation: pd.Series | float,
    output_gap: pd.Series | float,
    params: TaylorRuleParams,
) -> pd.Series | float:
    """
    Calcula la tasa de política nominal objetivo (r_o) según la Regla de
    Taylor generalizada.

        r_o = r_star + i_e + w_y * output_gap + w_pi * (i_e - i_o)

    Parameters
    ----------
    expected_inflation : serie o escalar con la inflación esperada (i_e),
        en %. Puede ser CPI YoY, PCE YoY, breakeven de mercado, o
        expectativas de encuesta (U. Michigan) — elegido por el PM.
    output_gap : serie o escalar con la brecha de producto (o de
        crecimiento), en %. Ver output_gap.py para los métodos de
        referencia.
    params : TaylorRuleParams con la calibración elegida por el PM.

    Returns
    -------
    Serie o escalar con la tasa objetivo implícita, en %.
    """
    inflation_gap = expected_inflation - params.inflation_target
    r_o = (
        params.r_star
        + expected_inflation
        + params.weight_output_gap * output_gap
        + params.weight_inflation_gap * inflation_gap
    )
    return r_o


def taylor_rate_timeseries(
    df: pd.DataFrame,
    expected_inflation_col: str,
    output_gap_series: pd.Series,
    params: TaylorRuleParams,
) -> pd.DataFrame:
    """
    Construye una serie de tiempo comparando:
      - la tasa objetivo según la Regla de Taylor calibrada
      - la tasa de fondos federales efectiva observada (FEDFUNDS)

    Útil para graficar en el dashboard la brecha histórica entre lo que
    "dictaría" la regla y lo que realmente hizo la Fed (señal cualitativa
    de postura: ¿hawkish/dovish respecto al fundamento?).
    """
    inflation = df[expected_inflation_col].dropna()
    gap_aligned = output_gap_series.reindex(inflation.index, method="ffill")
    joined = pd.concat([inflation, gap_aligned], axis=1).dropna()
    joined.columns = ["expected_inflation", "output_gap"]

    joined["taylor_rate"] = taylor_rate(
        joined["expected_inflation"], joined["output_gap"], params
    )

    fed_funds = df["FEDFUNDS"].dropna()
    joined = joined.join(fed_funds, how="left").ffill()
    joined["taylor_minus_fedfunds"] = joined["taylor_rate"] - joined["FEDFUNDS"]
    return joined


def qualitative_signal(taylor_minus_fedfunds: float, neutral_band: float = 0.25) -> str:
    """
    Traduce la brecha (Taylor - Fed Funds) en una señal cualitativa simple
    para el PM, con una banda neutral configurable (en puntos porcentuales).
    """
    if taylor_minus_fedfunds > neutral_band:
        return "Presión ALCISTA (regla sugiere tasas más altas que las actuales)"
    elif taylor_minus_fedfunds < -neutral_band:
        return "Presión BAJISTA (regla sugiere tasas más bajas que las actuales)"
    else:
        return "NEUTRAL (Fed Funds cercana a lo que sugiere la regla)"
