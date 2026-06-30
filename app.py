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
import numpy as np
import plotly.graph_objects as go

from src.fetch_fred import fetch_all_series, latest_value
from src.output_gap import all_output_gap_estimates
from src.r_star import all_r_star_estimates, fetch_rstar_nyfed, DEFAULT_RSTAR_FALLBACK
from src.taylor_rule import TaylorRuleParams, taylor_rate, taylor_rate_timeseries, qualitative_signal
from src.zscore_indicators import (
    CATEGORY_DEFINITIONS, TRANSFORM_LABELS,
    select_transforms_for_all_indicators, transformed_indicator_table,
    category_zscore, master_zscore, decompose_at_date,
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
    "ADS Index (Aruoba, Diebold & Scotti, 2009), con doble estandarización: "
    "cada indicador se transforma (la transformación se elige automáticamente "
    "mediante pruebas ADF/KPSS, no una regla única) y se estandariza; el "
    "compuesto ponderado de cada categoría se vuelve a estandarizar; y el "
    "compuesto entre categorías también se re-estandariza al final."
)


@st.cache_data(show_spinner="Corriendo pruebas ADF/KPSS por indicador...")
def _cached_transform_selection(data_fingerprint):
    return select_transforms_for_all_indicators(df)


transform_selections = _cached_transform_selection(df.index.max())

CATEGORY_LABELS = {"growth": "Crecimiento", "inflation": "Inflación", "employment": "Empleo"}

tab1, tab2, tab3 = st.tabs([
    "1. Datos y pesos", "2. Z-Scores individuales y descomposición", "3. Índice global",
])

# =============================================================================
# TAB 1 — Datos históricos, transformación elegida, pesos
# =============================================================================
with tab1:
    st.subheader("Transformación elegida por indicador (ADF/KPSS)")
    st.caption(
        "Cada indicador prueba 2-4 transformaciones candidatas según su tipo "
        "(tasa vs. índice con tendencia) y se elige la menos agresiva que "
        "resulte estadísticamente estacionaria — ver src/stationarity.py."
    )
    rows = []
    for category, indicators in CATEGORY_DEFINITIONS.items():
        for code, cfg in indicators.items():
            if code not in transform_selections:
                continue
            sel = transform_selections[code]
            chosen = sel["chosen_transform"]
            verdict = sel["results"].get(chosen, {}).get("verdict", "n/a")
            adf_p = sel["results"].get(chosen, {}).get("adf_pvalue")
            kpss_p = sel["results"].get(chosen, {}).get("kpss_pvalue")
            rows.append({
                "Categoría": CATEGORY_LABELS[category],
                "Indicador": cfg["label"],
                "Transformación elegida": TRANSFORM_LABELS.get(chosen, chosen),
                "Veredicto": verdict,
                "p-valor ADF": round(adf_p, 3) if adf_p is not None else None,
                "p-valor KPSS": round(kpss_p, 3) if kpss_p is not None else None,
            })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    st.subheader("Tabla histórica (mensual) de indicadores ya transformados")
    indicator_table = transformed_indicator_table(df, transform_selections)
    monthly_table = indicator_table.resample("ME").last()
    rename_map = {
        code: cfg["label"]
        for cat in CATEGORY_DEFINITIONS.values()
        for code, cfg in cat.items()
    }
    monthly_display = monthly_table.rename(columns=rename_map).round(2)
    st.dataframe(monthly_display.tail(24).sort_index(ascending=False), use_container_width=True)
    st.caption("Mostrando los últimos 24 meses; la tabla completa se usa internamente para el cálculo.")

    st.subheader("Pesos por categoría (deben sumar 1 dentro de cada eje)")
    st.caption(
        "Los pesos se renormalizan automáticamente para sumar 1 si no lo "
        "hacen — el valor normalizado real usado se muestra abajo de cada columna."
    )

    zscore_window_option = st.radio(
        "Ventana para calcular media/desviación estándar",
        options=["Muestra completa", "Ventana móvil (años)"],
        horizontal=True,
    )
    zscore_window_years = (
        st.slider("Años de la ventana móvil", 5, 20, 10)
        if zscore_window_option == "Ventana móvil (años)" else None
    )

    indicator_weights_by_category = {}
    cols = st.columns(3)
    for col, cat_key in zip(cols, ["growth", "inflation", "employment"]):
        with col:
            st.markdown(f"**{CATEGORY_LABELS[cat_key]}**")
            weights = {}
            for code, cfg in CATEGORY_DEFINITIONS[cat_key].items():
                weights[code] = st.number_input(
                    cfg["label"], min_value=0.0, max_value=1.0, value=round(1 / len(CATEGORY_DEFINITIONS[cat_key]), 2),
                    step=0.05, key=f"w_{cat_key}_{code}",
                )
            total = sum(weights.values())
            st.caption(f"Suma ingresada: {total:.2f}" + ("" if abs(total - 1.0) < 1e-6 else " → se renormaliza a 1.00"))
            indicator_weights_by_category[cat_key] = weights

    st.subheader("Pesos entre las 3 categorías para el Z-Score maestro")
    mc1, mc2, mc3 = st.columns(3)
    category_weights = {
        "growth": mc1.number_input("Peso Crecimiento", 0.0, 1.0, 0.2, 0.05),
        "inflation": mc2.number_input("Peso Inflación", 0.0, 1.0, 0.4, 0.05),
        "employment": mc3.number_input("Peso Empleo", 0.0, 1.0, 0.4, 0.05),
    }
    total_cat = sum(category_weights.values())
    st.caption(f"Suma ingresada: {total_cat:.2f}" + ("" if abs(total_cat - 1.0) < 1e-6 else " → se renormaliza a 1.00"))

    st.subheader("Evolución histórica por categoría")
    evo_cols = st.columns(3)
    for col, cat_key in zip(evo_cols, ["growth", "inflation", "employment"]):
        with col:
            fig = go.Figure()
            for code, cfg in CATEGORY_DEFINITIONS[cat_key].items():
                if code in indicator_table.columns:
                    fig.add_trace(go.Scatter(
                        x=indicator_table.index, y=indicator_table[code],
                        name=cfg["label"], opacity=0.8,
                    ))
            fig.update_layout(
                title=CATEGORY_LABELS[cat_key], showlegend=True,
                legend=dict(orientation="h", y=-0.3), height=320,
                margin=dict(t=40, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Dispersión: ¿se mueven juntos los indicadores?")
    scatter_cols = st.columns(2)
    all_codes = {code: cfg["label"] for cat in CATEGORY_DEFINITIONS.values() for code, cfg in cat.items()}
    with scatter_cols[0]:
        x_code = st.selectbox("Eje X", options=list(all_codes.keys()), format_func=lambda c: all_codes[c], index=0)
    with scatter_cols[1]:
        y_code = st.selectbox("Eje Y", options=list(all_codes.keys()), format_func=lambda c: all_codes[c], index=4)
    scatter_data = indicator_table[[x_code, y_code]].dropna()
    fig_scatter = go.Figure(go.Scatter(
        x=scatter_data[x_code], y=scatter_data[y_code], mode="markers",
        marker=dict(size=5, opacity=0.5, color=np.arange(len(scatter_data)), colorscale="Viridis"),
    ))
    fig_scatter.update_layout(
        xaxis_title=all_codes[x_code], yaxis_title=all_codes[y_code],
        title="Color = más reciente (amarillo) vs. más antiguo (morado)",
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

# ---------------------------------------------------------------------------
# Cálculo común (usado en tabs 2 y 3)
# ---------------------------------------------------------------------------
category_results = {
    cat: category_zscore(df, cat, transform_selections, indicator_weights_by_category[cat], zscore_window_years)
    for cat in ["growth", "inflation", "employment"]
}
master = master_zscore(category_results, category_weights)

# =============================================================================
# TAB 2 — Z-Scores individuales y descomposición por categoría
# =============================================================================
with tab2:
    st.subheader("Z-Score de cada categoría a través del tiempo")
    fig_cats = go.Figure()
    for cat_key in ["growth", "inflation", "employment"]:
        z = category_results[cat_key]["category_zscore"]
        fig_cats.add_trace(go.Scatter(x=z.index, y=z, name=CATEGORY_LABELS[cat_key]))
    fig_cats.add_hline(y=0, line_dash="dot", line_color="gray")
    fig_cats.update_layout(yaxis_title="Desviaciones estándar", legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig_cats, use_container_width=True)

    st.subheader("Descomposición: ¿qué indicador explica la desviación?")
    decomp_cols = st.columns(2)
    with decomp_cols[0]:
        selected_cat = st.selectbox(
            "Categoría a descomponer", options=["growth", "inflation", "employment"],
            format_func=lambda c: CATEGORY_LABELS[c],
        )
    valid_dates = category_results[selected_cat]["category_zscore"].dropna().index
    with decomp_cols[1]:
        selected_date = st.select_slider(
            "Fecha", options=list(valid_dates), value=valid_dates[-1],
            format_func=lambda d: d.strftime("%Y-%m-%d"),
        )

    res = category_results[selected_cat]
    decomp = decompose_at_date(res["contributions"], res["baseline"], selected_date)
    decomp_labels = {**{c: cfg["label"] for c, cfg in CATEGORY_DEFINITIONS[selected_cat].items()},
                      "(ajuste de media histórica)": "(ajuste de media histórica)"}
    decomp_named = decomp.rename(index=decomp_labels)

    fig_decomp = go.Figure(go.Bar(
        x=decomp_named.values, y=decomp_named.index, orientation="h",
        marker_color=["#c5663b" if v < 0 else "#1d6b5a" for v in decomp_named.values],
    ))
    fig_decomp.add_vline(x=0, line_color="gray")
    fig_decomp.update_layout(
        title=f"Aporte de cada indicador al Z-Score de {CATEGORY_LABELS[selected_cat]} "
              f"el {selected_date.strftime('%Y-%m-%d')} (suma = {decomp_named.sum():+.2f})",
        xaxis_title="Aporte en desviaciones estándar",
    )
    st.plotly_chart(fig_decomp, use_container_width=True)
    st.caption(
        f"Z-Score real de {CATEGORY_LABELS[selected_cat]} en esa fecha: "
        f"{res['category_zscore'].loc[selected_date]:+.2f} (debe coincidir con la suma de barras)."
    )

# =============================================================================
# TAB 3 — Índice global con bandas hawkish/dovish 100% libres
# =============================================================================
with tab3:
    st.subheader("Z-Score maestro a través del tiempo")

    st.caption(
        "Bandas hawkish/dovish: no existe un umbral 'correcto' validado para "
        "este caso de uso (la literatura, p. ej. Berge & Jordà 2011, deriva "
        "umbrales óptimos contra un target específico como recesiones NBER, "
        "no contra postura de política monetaria). Defínelos tú según tu "
        "propio criterio o backtesting."
    )
    band_cols = st.columns(4)
    dovish_extreme = band_cols[0].number_input("Dovish extremo (≤)", value=-1.5, step=0.1)
    dovish_moderate = band_cols[1].number_input("Dovish moderado (≤)", value=-0.5, step=0.1)
    hawkish_moderate = band_cols[2].number_input("Hawkish moderado (≥)", value=0.5, step=0.1)
    hawkish_extreme = band_cols[3].number_input("Hawkish extremo (≥)", value=1.5, step=0.1)

    master_series = master["master_zscore"]
    fig_master = go.Figure()
    fig_master.add_trace(go.Scatter(x=master_series.index, y=master_series, name="Z-Score maestro", line=dict(width=2, color="#1a1a17")))
    for cat_key in ["growth", "inflation", "employment"]:
        z = category_results[cat_key]["category_zscore"]
        fig_master.add_trace(go.Scatter(x=z.index, y=z, name=CATEGORY_LABELS[cat_key], opacity=0.35))
    fig_master.add_hline(y=dovish_extreme, line_dash="dash", line_color="#1d6b5a", annotation_text="Dovish extremo")
    fig_master.add_hline(y=dovish_moderate, line_dash="dot", line_color="#1d6b5a", annotation_text="Dovish moderado")
    fig_master.add_hline(y=hawkish_moderate, line_dash="dot", line_color="#c5663b", annotation_text="Hawkish moderado")
    fig_master.add_hline(y=hawkish_extreme, line_dash="dash", line_color="#c5663b", annotation_text="Hawkish extremo")
    fig_master.update_layout(yaxis_title="Desviaciones estándar", legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig_master, use_container_width=True)

    last_date_master = master_series.dropna().index[-1]
    last_value = master_series.loc[last_date_master]

    if last_value >= hawkish_extreme:
        signal = "HAWKISH EXTREMO"
    elif last_value >= hawkish_moderate:
        signal = "Hawkish moderado"
    elif last_value <= dovish_extreme:
        signal = "DOVISH EXTREMO"
    elif last_value <= dovish_moderate:
        signal = "Dovish moderado"
    else:
        signal = "Neutral"

    st.metric("Z-Score maestro (última fecha)", f"{last_value:+.2f}", help=f"Fecha: {last_date_master.strftime('%Y-%m-%d')}")
    st.info(f"**Señal según tus bandas:** {signal}")

    st.subheader("¿Por qué está ahí el índice? Descomposición por categoría")
    master_decomp = decompose_at_date(master["contributions"], master["baseline"], last_date_master)
    master_decomp_named = master_decomp.rename(index={**CATEGORY_LABELS, "(ajuste de media histórica)": "(ajuste de media histórica)"})

    fig_master_decomp = go.Figure(go.Bar(
        x=master_decomp_named.values, y=master_decomp_named.index, orientation="h",
        marker_color=["#c5663b" if v < 0 else "#1d6b5a" for v in master_decomp_named.values],
    ))
    fig_master_decomp.add_vline(x=0, line_color="gray")
    fig_master_decomp.update_layout(
        title=f"Aporte de cada categoría al Z-Score maestro (suma = {master_decomp_named.sum():+.2f})",
        xaxis_title="Aporte en desviaciones estándar",
    )
    st.plotly_chart(fig_master_decomp, use_container_width=True)

    dominant_cat = master_decomp.drop("(ajuste de media histórica)").abs().idxmax()
    st.markdown(f"**La categoría con mayor aporte (en valor absoluto) es: {CATEGORY_LABELS[dominant_cat]}.** Su propia descomposición por indicador:")

    res_dom = category_results[dominant_cat]
    decomp_dom = decompose_at_date(res_dom["contributions"], res_dom["baseline"], last_date_master)
    decomp_dom_named = decomp_dom.rename(index={
        **{c: cfg["label"] for c, cfg in CATEGORY_DEFINITIONS[dominant_cat].items()},
        "(ajuste de media histórica)": "(ajuste de media histórica)",
    })
    fig_dom = go.Figure(go.Bar(
        x=decomp_dom_named.values, y=decomp_dom_named.index, orientation="h",
        marker_color=["#c5663b" if v < 0 else "#1d6b5a" for v in decomp_dom_named.values],
    ))
    fig_dom.add_vline(x=0, line_color="gray")
    fig_dom.update_layout(xaxis_title="Aporte en desviaciones estándar", height=300)
    st.plotly_chart(fig_dom, use_container_width=True)

st.markdown("---")
st.caption(
    "Fuente de datos: FRED (Federal Reserve Bank of St. Louis). "
    "Este panel es un complemento cualitativo y no constituye asesoría de inversión."
)
