"""
Corrige en data/partidos/partidos.csv (la fuente de la tabla partidos que carga load_data) lo que encontro comparar.py
(seccion 2 de data/redescarga/informe.md):

  - 35 partidos definidos por penales guardados con el marcador de la tanda (la final de 2006 como 5:3): pasan a su
    marcador a los 90 o 120 minutos y la tanda queda en la columna nueva "penales" ("5:3")
  - Suecia - Austria de 1938 no se jugo (Austria fue anexionada y Suecia paso sin jugar): queda con jugado = 0. Se
    conserva la fila porque el fixture de la app la referencia como "Match 7"; el dataset de entrenamiento la descarta

Los partidos_<año>.csv no se tocan: son la salida historica del scraper, con nombres de epoca (Alemania Federal, Union
Sovietica), y partidos.csv tiene los mismos partidos en el mismo orden con los nombres normalizados de la BD. Por eso
cada correccion se ubica por (año, posicion dentro del año).

Se puede ejecutar varias veces: un partido ya corregido no vuelve a cambiar.

Uso (tras descargar con busca_partidos.py --todos):
    python experimentos/redescarga/corregir_penales.py
"""
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
PARTIDOS = RAIZ / "data" / "partidos"
REDESCARGA = RAIZ / "data" / "redescarga" / "partidos"

NO_JUGADOS = {(1938, "Suecia", "Austria")}


def correcciones():
  """
  ¿QUE HACE?
  Arma, desde la redescarga de Wikipedia, las correcciones de marcador por (año, posicion)

  ¿COMO LO HACE?
  0-) Para cada año compara fila a fila partidos_<año>.csv con su redescarga (mismo orden y cantidad; 1938 no tiene tandas y se salta)

  1-) Si la redescarga tiene tanda y el marcador guardado es el de la tanda, registra el marcador real y la tanda

  Returns:

    - diccionario (año, posicion) -> (marcador del guardado, marcador real "a:b", tanda "a:b")
  """
  salida = {}
  for ruta in sorted(REDESCARGA.glob("partidos_*.csv")):
    agno = int(ruta.stem.rsplit("_", 1)[-1])
    actual, nuevo = pd.read_csv(PARTIDOS / ruta.name, encoding="utf-8"), pd.read_csv(ruta, encoding="utf-8")
    if len(actual) != len(nuevo) or nuevo["penales_0"].isna().all():
      continue
    for posicion, (guardado, real) in enumerate(zip(actual["score"], nuevo.itertuples())):
      if pd.isna(real.penales_0):
        continue
      tanda = f"{int(real.penales_0)}:{int(real.penales_1)}"
      if guardado == tanda:
        salida[(agno, posicion)] = (guardado, f"{int(real.score_0)}:{int(real.score_1)}", tanda)
  return salida


def main():
  ruta = PARTIDOS / "partidos.csv"
  df = pd.read_csv(ruta, encoding="utf-8", dtype=str, keep_default_na=False)
  if "penales" not in df.columns:
    df["penales"] = ""
  if "jugado" not in df.columns:
    df["jugado"] = "1"
  posicion = df.groupby("year").cumcount()

  n = 0
  for (agno, pos), (guardado, real, tanda) in correcciones().items():
    fila = df.index[(df["year"] == str(agno)) & (posicion == pos)]
    assert len(fila) == 1, (agno, pos)
    i = fila[0]
    if df.at[i, "penales"] == tanda:   # ya corregido
      continue
    assert df.at[i, "score"] == guardado, (agno, pos, df.at[i, "home"], df.at[i, "away"], df.at[i, "score"], guardado)
    df.at[i, "score"], df.at[i, "penales"] = real, tanda
    n += 1
    print(f"{agno} {df.at[i, 'home']} - {df.at[i, 'away']}: {guardado} -> {real} ({tanda} p.)")

  for agno, home, away in NO_JUGADOS:
    df.loc[(df["year"] == str(agno)) & (df["home"] == home) & (df["away"] == away), "jugado"] = "0"

  df.to_csv(ruta, index=False, encoding="utf-8")
  print(f"partidos.csv: {n} marcadores corregidos, {int((df['jugado'] == '0').sum())} partido(s) no jugado(s)")


if __name__ == "__main__":
  main()
