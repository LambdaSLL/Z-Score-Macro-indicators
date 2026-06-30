"""
transform_config.py
--------------------
Carga/guarda la selección de transformaciones (ADF/KPSS) como un archivo
JSON estático, para que la app NO tenga que correr esas pruebas en cada
visita. La estacionariedad de una serie macro es una propiedad estructural
que cambia muy lentamente (si es que cambia), así que el flujo correcto
es: correr el análisis una vez (scripts/select_transforms.py, en Colab o
local), congelar el resultado en config/transform_selections.json, y que
la app de producción solo LEA ese archivo.

Si el archivo no existe (primera vez, o repo recién clonado sin haber
corrido el script), la app cae automáticamente a calcular las pruebas en
vivo, para que nunca se rompa — pero se le avisa al usuario que conviene
regenerar y commitear el archivo para que la app quede liviana.
"""

from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
CONFIG_FILE = CONFIG_DIR / "transform_selections.json"


def save_transform_config(selections: dict, fred_data_through: str) -> None:
    """Guarda la selección de transformaciones en config/transform_selections.json."""
    CONFIG_DIR.mkdir(exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "fred_data_through": fred_data_through,
        "selections": {
            code: {
                "chosen_transform": sel["chosen_transform"],
                "indicator_type": sel["indicator_type"],
                "verdict": sel["results"].get(sel["chosen_transform"], {}).get("verdict"),
                "adf_pvalue": sel["results"].get(sel["chosen_transform"], {}).get("adf_pvalue"),
                "kpss_pvalue": sel["results"].get(sel["chosen_transform"], {}).get("kpss_pvalue"),
            }
            for code, sel in selections.items()
        },
    }
    with open(CONFIG_FILE, "w") as f:
        json.dump(payload, f, indent=2)


def load_transform_config() -> dict | None:
    """
    Carga config/transform_selections.json si existe.

    Returns
    -------
    dict {code: {"chosen_transform": str, "indicator_type": str, ...}} listo
    para usar en lugar de select_transforms_for_all_indicators(), o None si
    el archivo no existe todavía (el llamador debe caer a cálculo en vivo).
    """
    if not CONFIG_FILE.exists():
        return None
    try:
        with open(CONFIG_FILE) as f:
            payload = json.load(f)
        return payload
    except Exception:
        return None
