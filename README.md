# Monitor cualitativo: Regla de Taylor + curva del Tesoro (FRED + Streamlit)

Complemento **cualitativo** a un modelo cuantitativo de la curva del Tesoro
(p. ej. Dynamic Nelson-Siegel / VAR con filtro de Kalman). En vez de imponer
una calibración fija de la Regla de Taylor, este dashboard:

1. Descarga datos públicos y gratuitos de **FRED**.
2. Calcula **varios métodos de referencia** para r* y la brecha de producto
   (output gap), mostrándolos lado a lado.
3. Deja toda la calibración final (r*, meta de inflación, pesos) en manos
   del usuario (PM) vía sliders — los métodos de referencia son solo un
   ancla, nunca un valor impuesto.
4. Compara la tasa objetivo resultante contra la Fed Funds efectiva y contra
   la curva del Tesoro (2 años, 10 años, spread 2s10s).

## Por qué esta arquitectura

Siguiendo el paper de Marín Paniagua & Chacón Vásquez (2026) — que usa la
Regla de Taylor como uno de los factores macro de su modelo DNSmacro — este
panel separa la señal cualitativa (Taylor + contexto macro) del modelo
cuantitativo (DNS/VAR), para que ambos se usen como chequeos cruzados
independientes al proyectar movimientos en la tasa a 2 años y 10 años.

## Instalación

```bash
git clone <tu-repo>
cd taylor-fred-dashboard
pip install -r requirements.txt
cp .env.example .env
# Edita .env y coloca tu FRED_API_KEY (gratuita en
# https://fredaccount.stlouisfed.org/apikeys)
```

## Uso local

```bash
streamlit run app.py
```

## Despliegue en Streamlit Community Cloud

1. Sube este repo a GitHub.
2. En streamlit.io/cloud, conecta el repo y selecciona `app.py` como
   archivo principal.
3. En "Secrets" de la app, agrega:
   ```toml
   FRED_API_KEY = "tu_api_key"
   ```
4. (Opcional) Activa el workflow de GitHub Actions
   (`.github/workflows/update_data.yml`) agregando el secret
   `FRED_API_KEY` en Settings → Secrets and variables → Actions del repo,
   para que el cache de datos (`data/fred_series.parquet`) se refresque
   automáticamente todos los días y la app no dependa de llamadas en vivo
   a la API en cada visita.

## Estructura

```
taylor-fred-dashboard/
├── app.py                      # Dashboard Streamlit
├── src/
│   ├── fetch_fred.py            # Descarga y cacheo de series FRED
│   ├── output_gap.py            # 4 métodos de referencia de output gap
│   ├── r_star.py                # 2 métodos automatizables + 1 manual (HLW)
│   └── taylor_rule.py           # Regla de Taylor generalizada y parametrizable
├── .github/workflows/
│   └── update_data.yml          # Cron diario de refresco de datos
├── requirements.txt
└── .env.example
```

## Métodos de referencia incluidos

**r\* (tasa real neutral):**
- TIPS a 10 años (`DFII10`) como proxy de mercado (FRED).
- Promedio móvil de crecimiento real de PIB a 10 años (FRED, heurística de
  "regla de oro").
- **Holston-Laubach-Williams / Laubach-Williams (NY Fed) — descarga en
  vivo con fallback automático.** `fetch_rstar_nyfed()` en `src/r_star.py`
  intenta descargar el archivo `.xlsx` oficial directamente de
  newyorkfed.org (primero HLW, luego LW, luego URLs legacy conocidas).
  Si la estructura del archivo cambia o la URL no responde, usa
  automáticamente un valor de respaldo fijo (1.46% por defecto,
  configurable). También se puede forzar un valor manual desde la barra
  lateral, que tiene prioridad absoluta. Un panel de "Estado de fuentes de
  datos" en la barra lateral muestra cuál de las tres fuentes (vivo /
  manual / respaldo) se está usando en cada momento.

**Brecha de producto:**
- CBO: PIB real vs. PIB potencial (`GDPC1` vs. `GDPPOT`).
- Filtro Hodrick-Prescott sobre el log del PIB real.
- Ley de Okun: brecha de desempleo (`NROU` - `UNRATE`) × coeficiente 2.0.
- Brecha de tasa de crecimiento (especificación exacta del Anexo A1 del
  paper de Marín & Chacón).

## Indicador Z-Score compuesto (Sección 4)

Implementa la misma lógica que el **Chicago Fed National Activity Index
(CFNAI)** y el **ADS Index** (Aruoba, Diebold & Scotti, 2009): cada serie
se transforma (nivel, YoY, o variación mensual), se estandariza (z-score,
con ventana móvil o muestra completa configurable) y se combina en un
promedio ponderado. A diferencia del CFNAI —que fija los pesos vía PCA—
aquí los pesos quedan **100% en manos del usuario**, tanto a nivel de cada
indicador individual dentro de su categoría (crecimiento, inflación,
empleo) como entre las 3 categorías para el Z-Score maestro final.

- **Crecimiento:** PIB real (YoY), producción industrial (YoY), ventas
  minoristas (YoY), utilización de capacidad.
- **Inflación:** CPI YoY, PCE YoY, breakevens 5y/10y, expectativas U.
  Michigan.
- **Empleo:** cambio mensual de nóminas, tasa de desempleo (invertida),
  solicitudes de desempleo (invertidas), salario promedio por hora.

## Limitaciones y supuestos

- No se reproduce el dato de Bloomberg ni los índices propietarios
  (FXHCUSJP/FXHCUSEU) del paper original; se sustituyen por series 100%
  públicas de FRED.
- El ISM Services/PMI no está disponible gratuitamente con el mismo
  detalle que en Bloomberg; se omite del cálculo de Taylor (aunque sí
  puede agregarse como serie adicional de contexto).
- El indicador Z-Score compuesto es una simplificación deliberada del
  CFNAI/ADS (3 categorías con pesos manuales en vez de PCA sobre 85
  series); su utilidad depende de la calidad de la calibración del PM,
  no de un ajuste estadístico óptimo.
- Este panel es informativo/cualitativo y **no constituye asesoría de
  inversión**. Las calibraciones (r*, meta de inflación, pesos) son
  responsabilidad exclusiva del usuario.
