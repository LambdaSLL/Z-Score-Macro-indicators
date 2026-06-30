"""
select_transforms.py
---------------------
Corre las pruebas ADF/KPSS para decidir la transformación de cada
indicador macro (crecimiento, inflación, empleo) y CONGELA el resultado en
config/transform_selections.json, para que la app de Streamlit no tenga
que recalcularlo en cada visita.

Pensado para correr ocasionalmente (no en cada commit, no en cada visita
de un usuario) — por ejemplo manualmente cada pocos meses, o cuando se
agregue/cambie un indicador. La estacionariedad de una serie macro es una
propiedad estructural que no cambia semana a semana.

USO EN GOOGLE COLAB:
    !git clone https://github.com/<tu-usuario>/Z-Score-Macro-indicators.git
    %cd Z-Score-Macro-indicators
    !pip install -r requirements.txt -q
    import os
    os.environ["FRED_API_KEY"] = "tu_api_key_aqui"
    !python scripts/select_transforms.py

Luego descarga (o copia el contenido de) config/transform_selections.json
y haz commit/push al repo desde tu máquina o directamente desde Colab con
git. La próxima vez que la app de Streamlit se despliegue, leerá ese
archivo en vez de correr las pruebas en vivo.

USO LOCAL:
    python scripts/select_transforms.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.fetch_fred import fetch_all_series
from src.zscore_indicators import select_transforms_for_all_indicators, TRANSFORM_LABELS, CATEGORY_DEFINITIONS
from src.transform_config import save_transform_config


def main() -> int:
    print("Descargando/leyendo datos de FRED...")
    df = fetch_all_series(force_refresh=True)
    data_through = str(df.dropna(how="all").index.max().date())
    print(f"Datos disponibles hasta: {data_through}\n")

    print("Corriendo pruebas ADF/KPSS por indicador (puede tardar uno o dos minutos)...\n")
    selections = select_transforms_for_all_indicators(df)

    print(f"{'Indicador':<15} {'Transformación elegida':<30} {'Veredicto':<15}")
    print("-" * 62)
    for category, indicators in CATEGORY_DEFINITIONS.items():
        for code, cfg in indicators.items():
            if code not in selections:
                continue
            sel = selections[code]
            chosen = sel["chosen_transform"]
            verdict = sel["results"].get(chosen, {}).get("verdict", "n/a")
            label = TRANSFORM_LABELS.get(chosen, chosen)
            print(f"{code:<15} {label:<30} {verdict:<15}")

    save_transform_config(selections, fred_data_through=data_through)
    print(f"\nGuardado en config/transform_selections.json.")
    print("Recuerda hacer commit y push de ese archivo para que la app de "
          "Streamlit lo use en vez de recalcular las pruebas en cada visita.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
