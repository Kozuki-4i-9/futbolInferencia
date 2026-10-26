"""
Partidos internacionales de selecciones (1872 - hoy) para entrenar con mas que eliminatorias y mundiales.

Fuente: github.com/martj42/international_results (results.csv), la misma de la que salio data/partidos/clasificaciones.csv.
Genera:

  - data/partidos/internacionales.csv: un partido por fila con fecha, agno, home, away, score_0, score_1, tournament,
    categoria y neutral, con los nombres en español de la BD
  - data/partidos/nombres_en_es.csv: la traduccion de nombres usada, para revisarla a mano

¿COMO TRADUCE LOS NOMBRES?
clasificaciones.csv (español) y los partidos "FIFA World Cup qualification" de results.csv (ingles) son la misma lista en
el mismo orden: se recorren año por año emparejando partidos con el mismo marcador y cada pareja vota la traduccion de
sus dos equipos. Las selecciones que nunca jugaron eliminatorias (Jersey, Pais Vasco, ...) conservan su nombre en ingles.

Uso:
    python etl/busca_internacionales.py            # descarga results.csv
    python etl/busca_internacionales.py --local    # usa data/redescarga/internacionales/results.csv
"""
import argparse
import urllib.request
from collections import Counter, defaultdict
from io import StringIO

import pandas as pd

from rutas import DATA, PARTIDOS

URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
LOCAL = DATA / "redescarga" / "internacionales" / "results.csv"
SALIDA = PARTIDOS / "internacionales.csv"
NOMBRES = PARTIDOS / "nombres_en_es.csv"

# torneos continentales de selecciones absolutas (el resto de torneos con nombre propio cae en "regional")
CONTINENTALES = {
  "UEFA Euro", "Copa América", "African Cup of Nations", "AFC Asian Cup", "Gold Cup", "CONCACAF Championship",
  "Oceania Nations Cup", "Confederations Cup",
}
# orden fijo: el modelo recibe la categoria como one-hot en este orden (ver fases/dataset.py)
CATEGORIAS = ["mundial", "eliminatoria_mundial", "continental", "eliminatoria_continental", "liga_naciones", "amistoso", "regional"]


def categoria(torneo):
  """Agrupa los 200+ nombres de torneo de la fuente en CATEGORIAS"""
  if torneo == "FIFA World Cup":
    return "mundial"
  if torneo == "FIFA World Cup qualification":
    return "eliminatoria_mundial"
  if torneo in CONTINENTALES:
    return "continental"
  if "Nations League" in torneo:
    return "liga_naciones" if "qualification" not in torneo else "eliminatoria_continental"
  if "qualification" in torneo:
    return "eliminatoria_continental"
  if torneo == "Friendly":
    return "amistoso"
  return "regional"


def traduccion(resultados, clasificaciones):
  """
  ¿QUE HACE?
  Aprende la traduccion ingles -> español de los nombres de selecciones alineando las eliminatorias de (resultados) con (clasificaciones)

  ¿COMO LO HACE?
  0-) Para cada año, recorre los partidos de (clasificaciones) en orden y busca hacia adelante (hasta 5 filas) el primero de (resultados) con el mismo marcador

  1-) Cada pareja encontrada suma un voto para home -> home y away -> away; gana la traduccion mas votada

  Returns:

    - diccionario nombre en ingles -> nombre en español
  """
  eliminatorias = resultados[resultados["tournament"] == "FIFA World Cup qualification"]
  votos = defaultdict(Counter)
  for agno, del_agno in clasificaciones.groupby("agno", sort=False):
    fuente = eliminatorias[eliminatorias["agno"] == agno].reset_index(drop=True)
    j = 0
    for fila in del_agno.itertuples():
      k = j
      while k < len(fuente) and not (fuente.home_score[k] == fila.score_0 and fuente.away_score[k] == fila.score_1):
        k += 1
      if k == len(fuente) or k - j > 5:
        continue
      votos[fuente.home_team[k]][fila.home] += 1
      votos[fuente.away_team[k]][fila.away] += 1
      j = k + 1
  return {ingles: v.most_common(1)[0][0] for ingles, v in votos.items()}


def main():
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--local", action="store_true", help=f"leer {LOCAL} en vez de descargar")
  args = parser.parse_args()

  texto = LOCAL.read_text(encoding="utf-8") if args.local else urllib.request.urlopen(URL, timeout=60).read().decode("utf-8")
  resultados = pd.read_csv(StringIO(texto), parse_dates=["date"]).dropna(subset=["home_score", "away_score"])
  resultados["agno"] = resultados["date"].dt.year
  clasificaciones = pd.read_csv(PARTIDOS / "clasificaciones.csv", encoding="utf-8")

  nombres = traduccion(resultados, clasificaciones)
  traducir = lambda serie: serie.map(lambda n: nombres.get(n, n))

  salida = pd.DataFrame({
    "fecha": resultados["date"].dt.strftime("%Y-%m-%d"),
    "agno": resultados["agno"],
    "home": traducir(resultados["home_team"]),
    "away": traducir(resultados["away_team"]),
    "score_0": resultados["home_score"].astype(int),
    "score_1": resultados["away_score"].astype(int),
    "tournament": resultados["tournament"],
    "categoria": resultados["tournament"].map(categoria),
    "neutral": resultados["neutral"].astype(int),
  }).sort_values("fecha", kind="stable").reset_index(drop=True)

  salida.to_csv(SALIDA, index_label="id", encoding="utf-8")
  pd.DataFrame(sorted(nombres.items()), columns=["ingles", "español"]).to_csv(NOMBRES, index=False, encoding="utf-8")

  traducidos = salida["home"].isin(nombres.values()) & salida["away"].isin(nombres.values())
  print(f"{len(salida)} partidos -> {SALIDA} ({traducidos.sum()} con los dos equipos en español)")
  print(salida["categoria"].value_counts().to_string())


if __name__ == "__main__":
  main()
