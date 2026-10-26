"""
Descarga el dataset publico de resultados internacionales (github.com/martj42/international_results, el mismo que
el "results0.csv" de Kaggle con que se armo data/partidos/clasificaciones.csv) y separa:

  - data/redescarga/internacionales/eliminatorias.csv: partidos "FIFA World Cup qualification"
  - data/redescarga/internacionales/mundiales.csv: partidos "FIFA World Cup", con la tanda de penales en columnas aparte

Guarda tambien los archivos originales (results.csv, shootouts.csv) para poder repetir el estudio sin red.
Los nombres de las selecciones quedan en ingles, como en la fuente.

Uso:
    python experimentos/redescarga/descarga_internacionales.py
"""
import urllib.request
from io import StringIO
from pathlib import Path

import pandas as pd

BASE = "https://raw.githubusercontent.com/martj42/international_results/master/"
SALIDA = Path(__file__).resolve().parents[2] / "data" / "redescarga" / "internacionales"


def descargar(nombre):
    """Descarga (nombre) de la fuente, lo guarda tal cual en SALIDA y lo devuelve como df"""
    texto = urllib.request.urlopen(BASE + nombre, timeout=60).read().decode("utf-8")
    (SALIDA / nombre).write_text(texto, encoding="utf-8")
    return pd.read_csv(StringIO(texto), parse_dates=["date"])


def main():
    SALIDA.mkdir(parents=True, exist_ok=True)
    resultados = descargar("results.csv")
    tandas = descargar("shootouts.csv")
    resultados["agno"] = resultados["date"].dt.year

    eliminatorias = resultados[resultados["tournament"] == "FIFA World Cup qualification"]
    eliminatorias.to_csv(SALIDA / "eliminatorias.csv", index=False, encoding="utf-8")

    # la tanda solo dice quien gano; se guarda el ganador junto al partido (90' o 120')
    mundiales = resultados[resultados["tournament"] == "FIFA World Cup"].merge(
        tandas[["date", "home_team", "away_team", "winner"]].rename(columns={"winner": "ganador_penales"}),
        on=["date", "home_team", "away_team"], how="left",
    )
    mundiales.to_csv(SALIDA / "mundiales.csv", index=False, encoding="utf-8")

    print(f"eliminatorias: {len(eliminatorias)} partidos ({eliminatorias['agno'].min()} - {eliminatorias['agno'].max()})")
    print(f"mundiales: {len(mundiales)} partidos, {mundiales['ganador_penales'].notna().sum()} definidos por penales")


if __name__ == "__main__":
    main()
