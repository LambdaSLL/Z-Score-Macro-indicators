"""
app.py
------
Dashboard cualitativo de monitoreo macro y Regla de Taylor para
complementar el modelo cuantitativo (DNS / VAR) de la curva del Tesoro.

Filosofía: la calibración de r*, meta de inflación y pesos queda 100% en
manos del PM (sliders). El dashboard solo muestra "métodos de referencia"
calculados con datos gratuitos de FRED, a modo de ancla — nunca impone un
valor.

Ejecutar con:
    streamlit run app.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from src.fetch_fred import fetch_all_series, latest_value
from src.output_gap import all_output_gap_estimates
from src.r_star import all_r_star_estimates, fetch_rstar_nyfed, DEFAULT_RSTAR_FALLBACK
from src.taylor_rule import TaylorRuleParams, taylor_rate, taylor_rate_timeseries, qualitative_signal
from src.zscore_indicators import (
    CATEGORY_DEFINITIONS, category_zscore, master_zscore, interpret_zscore
)

st.set_page_config(page_title="Taylor Rule & Curva del Tesoro", layout="wide")

st.title("Monitor cualitativo: Regla de Taylor y curva del Tesoro de EE.UU.")
st.caption(
    "Complemento cualitativo a modelos cuantitativos (DNS / VAR). "
    "Todos los parámetros de calibración son ajustables por el usuario; "
    "las tablas de 'métodos de referencia' son solo un punto de anclaje."
)

# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Datos")
    refresh = st.button("Refrescar datos de FRED")
    st.caption("Si no tienes FRED_API_KEY configurada, ver .env.example")

try:
    df = fetch_all_series(force_refresh=refresh)
except RuntimeError as e:
    st.error(str(e))
    st.stop()

st.sidebar.success(f"Datos al {df.dropna(how='all').index.max().date()}")

# ---------------------------------------------------------------------------
# r* del NY Fed: descarga en vivo (HLW/LW) con fallback, override manual,
# y panel de estado — mismo patrón que la app de Fair Value del usuario.
# ---------------------------------------------------------------------------
@st.cache_data(ttl=60 * 60 * 24)  # 24h: r* es trimestral y se mueve muy poco
def _cached_fetch_rstar_nyfed(manual_value: float | None):
    return fetch_rstar_nyfed(manual_override=manual_value)


with st.sidebar.expander("Estado de fuentes de datos: r* (NY Fed)", expanded=False):
    use_manual_rstar = st.checkbox("Forzar valor manual de r*", value=False)
    manual_rstar_value = None
    if use_manual_rstar:
        manual_rstar_value = st.number_input(
            "Valor manual de r* (%)", min_value=-2.0, max_value=4.0, value=DEFAULT_RSTAR_FALLBACK, step=0.05
        )

    rstar_result = _cached_fetch_rstar_nyfed(manual_rstar_value)

    if rstar_result["source"] == "manual":
        st.info(f"Usando valor MANUAL: r* = {rstar_result['value']:.2f}%")
    elif rstar_result["source"] == "live":
        st.success(
            f"Descarga EN VIVO exitosa desde NY Fed.\n\n"
            f"r* (último trimestre) = {rstar_result['value']:.2f}%\n\n"
            f"Fuente: {rstar_result['url']}"
        )
    else:
        st.warning(
            f"No se pudo descargar el archivo del NY Fed (URLs no accesibles "
            f"o estructura cambiada). Usando valor de RESPALDO fijo: "
            f"r* = {rstar_result['value']:.2f}%"
        )
    st.caption(
        "El r* HLW/LW se publica trimestralmente y cambia muy poco entre "
        "publicaciones, así que el impacto de usar el respaldo es bajo en "
        "el corto plazo."
    )

# ---------------------------------------------------------------------------
# Sidebar: calibración libre del PM
# ---------------------------------------------------------------------------
st.sidebar.header("Calibración de la Regla de Taylor (libre)")

r_star_pm = st.sidebar.slider(
    "r* — tasa real neutral (%)", min_value=-1.0, max_value=3.0,
    value=round(rstar_result["value"], 2), step=0.05,
    help="Sugerido a partir del panel 'Estado de fuentes de datos: r*' arriba; ajustable libremente.",
)
inflation_target_pm = st.sidebar.slider(
    "Meta de inflación (%)", min_value=0.0, max_value=4.0, value=2.0, step=0.1
)
weight_output_pm = st.sidebar.slider(
    "Peso brecha de producto (w_y)", min_value=0.0, max_value=1.5, value=0.5, step=0.05
)
weight_inflation_pm = st.sidebar.slider(
    "Peso brecha de inflación (w_pi)", min_value=0.0, max_value=1.5, value=0.5, step=0.05
)

inflation_measure = st.sidebar.selectbox(
    "Medida de inflación esperada a usar",
    options=["CPIAUCSL_YOY", "PCEPI_YOY", "T5YIE", "T10YIE", "MICH"],
    index=1,
    help="CPI/PCE YoY: interanual observado. T5YIE/T10YIE: breakeven de mercado. MICH: encuesta U. Michigan.",
)

output_gap_method = st.sidebar.selectbox(
    "Método de brecha de producto a usar",
    options=["output_gap_cbo", "output_gap_hp", "output_gap_okun", "output_growth_gap_paper"],
    index=0,
)

params = TaylorRuleParams(
    r_star=r_star_pm,
    inflation_target=inflation_target_pm,
    weight_output_gap=weight_output_pm,
    weight_inflation_gap=weight_inflation_pm,
)

# ---------------------------------------------------------------------------
# Sección 1: métodos de referencia (solo informativos)
# ---------------------------------------------------------------------------
st.header("1. Métodos de referencia (ancla, no imposición)")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Referencias para r*")
    r_star_table = all_r_star_estimates(df)
    last_r_star = r_star_table.dropna(how="all").iloc[-1]
    ref_df_r = pd.DataFrame({
        "Método": [
            "TIPS 10 años (mercado, diario)",
            "Crecimiento real de tendencia (10 años)",
            f"Holston-Laubach-Williams (NY Fed, {rstar_result['source']})",
        ],
        "Valor más reciente (%)": [
            round(last_r_star.get("r_star_tips10", float("nan")), 2),
            round(last_r_star.get("r_star_long_run_growth", float("nan")), 2),
            round(rstar_result["value"], 2),
        ],
        "Notas": [
            "Incluye prima de plazo/liquidez",
            "Heurística de regla de oro",
            "Ver panel 'Estado de fuentes de datos' en la barra lateral",
        ],
    })
    st.dataframe(ref_df_r, hide_index=True, use_container_width=True)
    st.caption(f"Tu calibración actual: r* = {r_star_pm}%")

with col2:
    st.subheader("Referencias para la brecha de producto")
    gap_table = all_output_gap_estimates(df)
    last_gap = gap_table.dropna(how="all").iloc[-1]
    ref_df_g = pd.DataFrame({
        "Método": [
            "CBO (PIB real vs. potencial)",
            "Filtro Hodrick-Prescott",
            "Ley de Okun (brecha de desempleo)",
            "Brecha de crecimiento (paper Marín & Chacón)",
        ],
        "Valor más reciente (%)": [
            round(last_gap.get("output_gap_cbo", float("nan")), 2),
            round(last_gap.get("output_gap_hp", float("nan")), 2),
            round(last_gap.get("output_gap_okun", float("nan")), 2),
            round(last_gap.get("output_growth_gap_paper", float("nan")), 2),
        ],
        "Notas": [
            "Oficial, pero revisado con rezago",
            "Sensible al final de la muestra",
            "Coeficiente de Okun fijo en 2.0",
            "Brecha de TASA de crecimiento, no de nivel",
        ],
    })
    st.dataframe(ref_df_g, hide_index=True, use_container_width=True)

    selected_gap_value = last_gap.get(output_gap_method, float("nan"))
    st.caption(f"Brecha seleccionada para el cálculo ({output_gap_method}): {selected_gap_value:.2f}%")

# ---------------------------------------------------------------------------
# Sección 2: resultado de la Regla de Taylor calibrada
# ---------------------------------------------------------------------------
st.header("2. Regla de Taylor con tu calibración")

current_inflation = latest_value(df, inflation_measure)
current_fedfunds = latest_value(df, "FEDFUNDS")

current_taylor = taylor_rate(current_inflation, selected_gap_value, params)
gap_vs_fed = current_taylor - current_fedfunds
signal = qualitative_signal(gap_vs_fed)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Inflación esperada usada", f"{current_inflation:.2f}%")
m2.metric("Fed Funds efectiva actual", f"{current_fedfunds:.2f}%")
m3.metric("Tasa objetivo (Regla de Taylor)", f"{current_taylor:.2f}%")
m4.metric("Brecha (Taylor - Fed Funds)", f"{gap_vs_fed:+.2f} pp")

if gap_vs_fed > 0.25:
    st.warning(f"**Señal cualitativa:** {signal}")
elif gap_vs_fed < -0.25:
    st.info(f"**Señal cualitativa:** {signal}")
else:
    st.success(f"**Señal cualitativa:** {signal}")

# Serie histórica Taylor vs Fed Funds
gap_series_for_history = gap_table[output_gap_method].dropna()
history = taylor_rate_timeseries(df, inflation_measure, gap_series_for_history, params)

fig_taylor = go.Figure()
fig_taylor.add_trace(go.Scatter(x=history.index, y=history["taylor_rate"], name="Regla de Taylor (calibrada)"))
fig_taylor.add_trace(go.Scatter(x=history.index, y=history["FEDFUNDS"], name="Fed Funds efectiva"))
fig_taylor.update_layout(
    title="Tasa objetivo (Regla de Taylor) vs. Fed Funds efectiva",
    yaxis_title="%", xaxis_title="Fecha", legend=dict(orientation="h", y=-0.2),
)
st.plotly_chart(fig_taylor, use_container_width=True)

# ---------------------------------------------------------------------------
# Sección 3: monitoreo de la curva del Tesoro (2y, 10y, spread)
# ---------------------------------------------------------------------------
st.header("3. Monitoreo de la curva del Tesoro (2 y 10 años)")

c1, c2, c3 = st.columns(3)
c1.metric("UST 2Y", f"{latest_value(df, 'DGS2'):.2f}%")
c2.metric("UST 10Y", f"{latest_value(df, 'DGS10'):.2f}%")
c3.metric("Spread 10Y-2Y", f"{latest_value(df, 'T10Y2Y'):+.2f} pp")

fig_curve = go.Figure()
fig_curve.add_trace(go.Scatter(x=df.index, y=df["DGS2"], name="UST 2Y"))
fig_curve.add_trace(go.Scatter(x=df.index, y=df["DGS10"], name="UST 10Y"))
fig_curve.update_layout(title="Rendimientos UST 2Y y 10Y", yaxis_title="%", xaxis_title="Fecha")
st.plotly_chart(fig_curve, use_container_width=True)

fig_spread = go.Figure()
fig_spread.add_trace(go.Scatter(x=df.index, y=df["T10Y2Y"], name="Spread 10Y-2Y", fill="tozeroy"))
fig_spread.add_hline(y=0, line_dash="dash", line_color="red")
fig_spread.update_layout(title="Spread 10Y-2Y (pendiente de la curva)", yaxis_title="pp", xaxis_title="Fecha")
st.plotly_chart(fig_spread, use_container_width=True)

st.markdown("---")
st.caption(
    "Fuente de datos: FRED (Federal Reserve Bank of St. Louis). "
    "Este panel es un complemento cualitativo y no constituye asesoría de inversión."
)

# ---------------------------------------------------------------------------
# Sección 4: indicador Z-Score compuesto (crecimiento / inflación / empleo)
# ---------------------------------------------------------------------------
st.header("4. Indicador Z-Score compuesto de desviación macro")
st.caption(
    "Misma lógica que el Chicago Fed National Activity Index (CFNAI) y el "
    "ADS Index (Aruoba, Diebold & Scotti, 2009): cada serie se estandariza "
    "(z-score) y se combina en un promedio ponderado. A diferencia del "
    "CFNAI, aquí los pesos son 100% definidos por el usuario, no por PCA."
)

with st.expander("Configurar ventana de normalización y pesos", expanded=False):
    zscore_window_option = st.radio(
        "Ventana para calcular media/desviación estándar",
        options=["Muestra completa", "Ventana móvil (años)"],
        horizontal=True,
    )
    if zscore_window_option == "Ventana móvil (años)":
        zscore_window_years = st.slider("Años de la ventana móvil", 5, 20, 10)
    else:
        zscore_window_years = None

    st.markdown("**Pesos individuales dentro de cada categoría** (se renormalizan automáticamente)")
    category_weights = {}
    indicator_weights_by_category = {}

    cols = st.columns(3)
    for col, (cat_key, cat_label) in zip(
        cols, [("growth", "Crecimiento"), ("inflation", "Inflación"), ("employment", "Empleo")]
    ):
        with col:
            st.markdown(f"*{cat_label}*")
            weights = {}
            for code, cfg in CATEGORY_DEFINITIONS[cat_key].items():
                weights[code] = st.number_input(
                    cfg["label"], min_value=0.0, max_value=5.0, value=1.0, step=0.25,
                    key=f"w_{cat_key}_{code}",
                )
            indicator_weights_by_category[cat_key] = weights

    st.markdown("**Pesos entre las 3 categorías para el Z-Score maestro**")
    mc1, mc2, mc3 = st.columns(3)
    category_weights["growth"] = mc1.number_input("Peso Crecimiento", 0.0, 5.0, 1.0, 0.25)
    category_weights["inflation"] = mc2.number_input("Peso Inflación", 0.0, 5.0, 1.0, 0.25)
    category_weights["employment"] = mc3.number_input("Peso Empleo", 0.0, 5.0, 1.0, 0.25)

growth_z, growth_table = category_zscore(
    df, "growth", indicator_weights_by_category["growth"], zscore_window_years
)
inflation_z, inflation_table = category_zscore(
    df, "inflation", indicator_weights_by_category["inflation"], zscore_window_years
)
employment_z, employment_table = category_zscore(
    df, "employment", indicator_weights_by_category["employment"], zscore_window_years
)
master_z = master_zscore(growth_z, inflation_z, employment_z, category_weights)

z1, z2, z3, z4 = st.columns(4)
z1.metric("Z-Score Crecimiento", f"{growth_z.dropna().iloc[-1]:+.2f}")
z2.metric("Z-Score Inflación", f"{inflation_z.dropna().iloc[-1]:+.2f}")
z3.metric("Z-Score Empleo", f"{employment_z.dropna().iloc[-1]:+.2f}")
z4.metric("Z-Score MAESTRO", f"{master_z.dropna().iloc[-1]:+.2f}")

st.info(f"**Lectura cualitativa:** {interpret_zscore(master_z.dropna().iloc[-1])}")

fig_master = go.Figure()
fig_master.add_trace(go.Scatter(x=master_z.index, y=master_z, name="Z-Score maestro", line=dict(width=2)))
fig_master.add_trace(go.Scatter(x=growth_z.index, y=growth_z, name="Crecimiento", opacity=0.5))
fig_master.add_trace(go.Scatter(x=inflation_z.index, y=inflation_z, name="Inflación", opacity=0.5))
fig_master.add_trace(go.Scatter(x=employment_z.index, y=employment_z, name="Empleo", opacity=0.5))
fig_master.add_hline(y=0, line_dash="dot", line_color="gray")
fig_master.add_hline(y=1.5, line_dash="dash", line_color="red", opacity=0.4)
fig_master.add_hline(y=-1.5, line_dash="dash", line_color="red", opacity=0.4)
fig_master.update_layout(
    title="Z-Score maestro y sub-índices a través del tiempo",
    yaxis_title="Desviaciones estándar", xaxis_title="Fecha",
    legend=dict(orientation="h", y=-0.2),
)
st.plotly_chart(fig_master, use_container_width=True)

with st.expander("Ver detalle de Z-Scores individuales por indicador"):
    detail_cols = st.columns(3)
    for col, (label, table) in zip(
        detail_cols,
        [("Crecimiento", growth_table), ("Inflación", inflation_table), ("Empleo", employment_table)],
    ):
        with col:
            st.markdown(f"**{label}**")
            last_row = table.dropna(how="all").iloc[-1].round(2)
            st.dataframe(last_row.rename("Z-Score"), use_container_width=True)

