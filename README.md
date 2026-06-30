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
│   ├── r_star.py                # Descarga en vivo NY Fed + fallback + manual
│   ├── taylor_rule.py           # Regla de Taylor generalizada y parametrizable
│   ├── stationarity.py          # Pruebas ADF/KPSS y selección de transformación
│   ├── transform_config.py      # Carga/guarda config/transform_selections.json
│   └── zscore_indicators.py     # Z-Score compuesto con doble estandarización
├── scripts/
│   └── select_transforms.py     # Script offline (Colab/local) para congelar transformaciones
├── config/
│   └── transform_selections.json  # Generado por el script — no se commitea hasta correrlo
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

## Indicador Z-Score compuesto (Sección 4 — 3 pestañas)

Implementa la misma lógica que el **Chicago Fed National Activity Index
(CFNAI)** y el **ADS Index** (Aruoba, Diebold & Scotti, 2009), con **doble
estandarización**:

1. Cada indicador se transforma según la transformación que las pruebas
   **ADF/KPSS** determinen como la más adecuada para esa serie (siguiendo
   el espíritu de McCracken & Ng, 2016, FRED-MD: no toda la serie se
   transforma igual — algunas son estacionarias en nivel, otras necesitan
   variación interanual o diferencia).
2. Los Z-Scores individuales de cada categoría se combinan en un promedio
   ponderado (pesos del PM, deben sumar 1 dentro de cada eje) y **ese
   compuesto se vuelve a estandarizar** — un promedio ponderado de
   Z-Scores correlacionados no tiene automáticamente desviación estándar
   1, así que CFNAI también re-normaliza el compuesto final.
3. Los 3 Z-Scores de categoría se combinan con un peso libre entre ellos
   y **también se re-estandarizan**, dando el Z-Score maestro.
4. Cada nivel de combinación tiene una **descomposición exacta**: el
   aporte de cada componente se calcula de forma que la suma de las
   barras reproduce exactamente el Z-Score final.

### Transformaciones: congeladas vs. en vivo

La estacionariedad de una serie macro es una propiedad estructural que no
cambia semana a semana, así que correr ADF/KPSS en cada visita del
dashboard es trabajo innecesario. El flujo recomendado:

1. Corre `python scripts/select_transforms.py` ocasionalmente (en Google
   Colab o local, no en cada commit) — descarga datos de FRED, corre
   ADF/KPSS por indicador, e imprime un resumen.
2. Esto genera/actualiza `config/transform_selections.json`.
3. Haz commit y push de ese archivo.
4. La app de Streamlit detecta el archivo y lo usa directamente, sin
   recalcular nada — la pestaña 1 muestra un aviso verde confirmando que
   está usando la versión congelada y la fecha en que se generó.

Si el archivo no existe (primer despliegue, o repo recién clonado), la
app cae automáticamente a calcular las pruebas en vivo en esa sesión (más
lento, con un aviso amarillo sugiriendo correr el script), para que nunca
se rompa por falta del archivo. En pruebas locales, con config congelado
la app cargó en ~2.5s vs. ~12s en modo de cálculo en vivo.

**Uso en Google Colab** (ver docstring completo en `scripts/select_transforms.py`):
```python
!git clone https://github.com/<tu-usuario>/Z-Score-Macro-indicators.git
%cd Z-Score-Macro-indicators
!pip install -r requirements.txt -q
import os
os.environ["FRED_API_KEY"] = "tu_api_key_aqui"
!python scripts/select_transforms.py
# luego: commit y push de config/transform_selections.json
```

### Las 3 pestañas

**Pestaña 1 — Datos y pesos:** estado del config de transformaciones
(congelado vs. en vivo), tabla de transformación elegida por indicador
(con p-valores ADF/KPSS), tabla histórica mensual ya transformada, inputs
de pesos por categoría y entre categorías, gráficos de evolución por
categoría y de dispersión entre pares de indicadores.

**Pestaña 2 — Z-Scores individuales y descomposición:** evolución de los
3 Z-Scores de categoría, y descomposición en gráfico de barras horizontal
de qué indicador explica la desviación de una categoría en una fecha
elegida.

**Pestaña 3 — Índice global:** Z-Score maestro a través del tiempo, con
bandas hawkish/dovish (extremo/moderado) **100% ajustables por el
usuario, sin un default "correcto" sugerido** — la literatura (p. ej.
Berge & Jordà, 2011) deriva umbrales óptimos contra un target específico
como recesiones NBER, no existe un equivalente validado para postura de
política monetaria. Incluye descomposición por categoría y, para la
categoría dominante, descomposición adicional por indicador.

- **Crecimiento:** PIB real, producción industrial, ventas minoristas,
  utilización de capacidad.
- **Inflación:** CPI YoY, PCE YoY, breakevens 5y/10y, expectativas U.
  Michigan.
- **Empleo:** nóminas no agrícolas, tasa de desempleo (invertida),
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
