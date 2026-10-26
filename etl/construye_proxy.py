"""
Arma data/proxy_desempegno/metricas_abstractas_<año>.csv (lo que carga load_data en la tabla proxy_desempegno): una fila
por seleccion y mundial con sus partidos (PJ), VI, SOT, PKATT y PKATTALLOW, de donde salen las puntuaciones por posicion
(gkps, mds, mos, mms) de cada equipo. PJ es el denominador de las tasas por partido del historico.

De donde sale cada metrica:

  - VI (porterias a cero): de los marcadores de data/partidos/partidos.csv, todos los años (exacto, no hace falta fbref)
  - PKATT / PKATTALLOW (penales a favor / en contra): de fbref (busca_metricas.py) donde los tenga, desde 1998; antes,
    tasa media de penales por partido de los años con dato x partidos jugados. pk_real = 1 si es dato real
  - SOT (tiros a puerta): de fbref desde 2018; antes, goles / tasa de goles por tiro a puerta de los años con dato.
    sot_real = 1 si es dato real

Si todavia no se ejecuto busca_metricas.py (data/proxy_desempegno/fbref/ vacio), las tasas usan los valores de
referencia TASA_PK_REFERENCIA y TASA_GOL_SOT_REFERENCIA y todo lo que no es VI queda imputado (pk_real = sot_real = 0).

Uso:
    python etl/construye_proxy.py
"""
import pandas as pd

from rutas import PARTIDOS, PROXY_DESEMPEGNO

FBREF = PROXY_DESEMPEGNO / "fbref"

# valores de referencia mientras no haya datos reales: ~0,12 penales por equipo y partido en mundiales previos al VAR,
# ~1 gol cada 3 tiros a puerta; se reemplazan por las tasas medidas en cuanto existan los csv de fbref
TASA_PK_REFERENCIA = 0.12
TASA_GOL_SOT_REFERENCIA = 0.33

# nombres de fbref -> nombre en la BD (si no esta aqui se prueba la traduccion de martj42 y el nombre tal cual)
TRADUCCION_FBREF = {
  "West Germany": "Alemania", "Germany DR": "Alemania Democrática", "East Germany": "Alemania Democrática",
  "Soviet Union": "Rusia", "IR Iran": "Irán", "Korea Republic": "Corea del Sur", "Korea DPR": "Corea del Norte",
  "USA": "Estados Unidos", "Rep. of Ireland": "Irlanda", "Côte d'Ivoire": "Costa de Marfil", "China PR": "China",
  "UAE": "Emiratos Árabes Unidos", "Czechia": "República Checa", "Czech Republic": "República Checa",
  "Serbia and Montenegro": "Serbia", "FR Yugoslavia": "Serbia", "Congo DR": "República Democrática del Congo",
  "Zaire": "República Democrática del Congo", "Dutch East Indies": "Indias Orientales Neerlandesas",
  "Türkiye": "Turquía", "Turkey": "Turquía", "Bosnia & Herz'na": "Bosnia y Herzegovina", "Qatar": "Catar",
  "Trinidad & Tobago": "Trinidad y Tobago",
}
# un mismo nombre de fbref que la BD guarda distinto segun el año (Yugoslavia de 1998 = "Serbia")
ALTERNATIVAS = {"Yugoslavia": ["Yugoslavia", "Serbia"]}


def estadisticas_por_mundial(partidos):
  """
  ¿QUE HACE?
  Por seleccion y mundial: partidos jugados, goles a favor, goles en contra y porterias a cero (VI), desde los marcadores

  Args:

    - partidos: df de partidos.csv (home, score, away, year, jugado)

  Returns:

    - df con agno, pais, PJ, GF, GC, VI
  """
  jugados = partidos[partidos["jugado"].astype(int) == 1]
  goles = jugados["score"].str.extract(r"^(\d+):(\d+)").astype(int)
  lados = pd.concat([
    pd.DataFrame({"agno": jugados["year"], "pais": jugados["home"], "GF": goles[0], "GC": goles[1]}),
    pd.DataFrame({"agno": jugados["year"], "pais": jugados["away"], "GF": goles[1], "GC": goles[0]}),
  ])
  lados["VI"] = (lados["GC"] == 0).astype(int)
  lados["PJ"] = 1
  return lados.groupby(["agno", "pais"], as_index=False)[["PJ", "GF", "GC", "VI"]].sum()


def traducir(equipo, equipos_del_agno, nombres_martj42):
  """Nombre de fbref (equipo) -> nombre de la BD entre (equipos_del_agno); None si ninguna traduccion encaja"""
  candidatos = ALTERNATIVAS.get(equipo, []) + [TRADUCCION_FBREF.get(equipo), nombres_martj42.get(equipo), equipo]
  return next((c for c in candidatos if c in equipos_del_agno), None)


def leer_fbref(estadisticas, nombres_martj42):
  """
  ¿QUE HACE?
  Lee los csv de busca_metricas.py y los deja con los nombres de la BD, avisando de los equipos que no se pudieron traducir

  Returns:

    - df con agno, pais, SOT, PKATT, PKATTALLOW (NaN donde fbref no tiene el dato), vacio si no hay csv
  """
  tablas = []
  for ruta in sorted(FBREF.glob("fbref_*.csv")):
    agno = int(ruta.stem.rsplit("_", 1)[-1])
    df = pd.read_csv(ruta, encoding="utf-8")
    equipos = set(estadisticas.loc[estadisticas["agno"] == agno, "pais"])
    df["pais"] = df["equipo"].map(lambda e: traducir(e, equipos, nombres_martj42))
    sin_traducir = sorted(df.loc[df["pais"].isna(), "equipo"])
    if sin_traducir:
      print(f"¡Atención! {ruta.name}: sin traduccion {sin_traducir} (agregalos a TRADUCCION_FBREF)")
    tablas.append(df.dropna(subset=["pais"]).assign(agno=agno)[["agno", "pais", "SOT", "PKATT", "PKATTALLOW"]])
  return pd.concat(tablas, ignore_index=True) if tablas else pd.DataFrame(columns=["agno", "pais", "SOT", "PKATT", "PKATTALLOW"])


def construir_proxy(estadisticas, fbref):
  """
  ¿QUE HACE?
  Une las estadisticas de los marcadores con lo real de fbref e imputa lo que falta

  ¿COMO LO HACE?
  0-) Une por (agno, pais); un dato de fbref es real si no es NaN

  1-) Tasa de penales = penales reales (a favor y en contra) / (2 x partidos de esas filas); tasa de goles por tiro a puerta = goles / SOT reales. Sin datos reales usa las de referencia

  2-) Donde no hay dato real: PKATT = PKATTALLOW = tasa de penales x PJ; SOT = GF / tasa de goles por tiro a puerta

  Returns:

    - (df con agno, pais, PJ, VI, SOT, PKATT, PKATTALLOW, pk_real, sot_real; diccionario con las tasas usadas)
  """
  df = estadisticas.merge(fbref, on=["agno", "pais"], how="left")
  pk_real = df["PKATT"].notna() & df["PKATTALLOW"].notna()
  sot_real = df["SOT"].notna()

  tasa_pk = (df.loc[pk_real, "PKATT"].sum() + df.loc[pk_real, "PKATTALLOW"].sum()) / (2 * df.loc[pk_real, "PJ"].sum()) if pk_real.any() else TASA_PK_REFERENCIA
  tasa_sot = df.loc[sot_real, "GF"].sum() / df.loc[sot_real, "SOT"].sum() if sot_real.any() else TASA_GOL_SOT_REFERENCIA

  df.loc[~pk_real, "PKATT"] = (df.loc[~pk_real, "PJ"] * tasa_pk).round(2)
  df.loc[~pk_real, "PKATTALLOW"] = df.loc[~pk_real, "PKATT"]
  df.loc[~sot_real, "SOT"] = (df.loc[~sot_real, "GF"] / tasa_sot).round(1)
  df["pk_real"], df["sot_real"] = pk_real.astype(int), sot_real.astype(int)
  return df[["agno", "pais", "PJ", "VI", "SOT", "PKATT", "PKATTALLOW", "pk_real", "sot_real"]], {"penales_por_partido": tasa_pk, "goles_por_tiro_a_puerta": tasa_sot}


def main():
  partidos = pd.read_csv(PARTIDOS / "partidos.csv", encoding="utf-8")
  nombres_martj42 = dict(pd.read_csv(PARTIDOS / "nombres_en_es.csv", encoding="utf-8").itertuples(index=False))
  estadisticas = estadisticas_por_mundial(partidos)
  proxy, tasas = construir_proxy(estadisticas, leer_fbref(estadisticas, nombres_martj42))

  PROXY_DESEMPEGNO.mkdir(parents=True, exist_ok=True)
  for agno, del_agno in proxy.groupby("agno"):
    del_agno.drop(columns="agno").to_csv(PROXY_DESEMPEGNO / f"metricas_abstractas_{agno}.csv", index=False, encoding="utf-8")

  print(f"{len(proxy)} selecciones x mundial en {proxy['agno'].nunique()} mundiales -> {PROXY_DESEMPEGNO}")
  print(f"datos reales: penales {proxy['pk_real'].sum()} filas, tiros a puerta {proxy['sot_real'].sum()} filas")
  print(f"tasas: {tasas['penales_por_partido']:.3f} penales por equipo y partido, {tasas['goles_por_tiro_a_puerta']:.3f} goles por tiro a puerta")


if __name__ == "__main__":
  main()
