"""
Contexto de cada partido calculado desde la tabla internacionales (todos los partidos de selecciones con fecha):

  - rating Elo de cada seleccion (reglas de eloratings.net): antes de cada partido y al inicio de cada mundial
  - confederacion de cada seleccion en una epoca y confederacion del anfitrion de cada mundial
  - dias de descanso desde el partido anterior de cada seleccion

Lo usan "construir_dataset" (entrenamiento) y la simulacion de la app (con "contexto_de_mundiales", cacheado: el Elo
recorre los ~50.000 partidos una sola vez por proceso), asi los dos ven exactamente los mismos valores.
"""
from collections import Counter, defaultdict
from functools import lru_cache

import numpy as np
import pandas as pd

ELO_INICIAL = 1500.0
VENTAJA_LOCAL = 100.0
# peso de cada tipo de partido (eloratings.net: 60 mundial, 50 continental, 40 eliminatorias, 30 otros torneos, 20 amistosos)
K_POR_CATEGORIA = {"mundial": 60, "continental": 50, "eliminatoria_mundial": 40, "eliminatoria_continental": 40,
                   "liga_naciones": 40, "regional": 30, "amistoso": 20}
# torneo continental -> confederacion (por prefijo del nombre en la fuente)
PREFIJOS_CONFEDERACION = [
  ("UEFA", "UEFA"), ("African Cup of Nations", "CAF"), ("COSAFA", "CAF"), ("Copa América", "CONMEBOL"),
  ("AFC", "AFC"), ("AFF", "AFC"), ("EAFF", "AFC"), ("ASEAN", "AFC"), ("Gold Cup", "CONCACAF"), ("CONCACAF", "CONCACAF"),
  ("CFU", "CONCACAF"), ("Oceania Nations Cup", "OFC"),
]
VENTANA_CONFEDERACION = 12  # años hacia atras para decidir la confederacion de una seleccion (Australia: OFC -> AFC)
DESCANSO_MAXIMO = 60        # dias; mas que eso es "llega descansado" igual


def multiplicador_goles(diferencia):
  """Multiplicador de eloratings.net por diferencia de goles"""
  n = abs(diferencia)
  return 1.0 if n <= 1 else 1.5 if n == 2 else (11 + n) / 8


def calcular_elo(internacionales):
  """
  ¿QUE HACE?
  Recorre (internacionales) en orden de fecha y calcula el Elo de cada seleccion antes de cada partido

  ¿COMO LO HACE?
  0-) Todas arrancan en ELO_INICIAL; el local suma VENTAJA_LOCAL a su rating para el resultado esperado si no es campo neutral

  1-) Tras cada partido: R += K * G * (resultado - esperado), con K segun la categoria y G segun la diferencia de goles

  Returns:

    - (elo_0, elo_1, despues_0, despues_1): arrays con el Elo de home y away antes y despues de cada partido, en el orden de las filas de (internacionales)
  """
  orden = np.argsort(internacionales["fecha"].to_numpy(), kind="stable")
  rating = defaultdict(lambda: ELO_INICIAL)
  elo_0, elo_1, despues_0, despues_1 = (np.zeros(len(internacionales)) for _ in range(4))
  home, away = internacionales["home"].to_numpy(), internacionales["away"].to_numpy()
  g0, g1 = internacionales["score_0"].to_numpy(), internacionales["score_1"].to_numpy()
  neutral, categoria = internacionales["neutral"].to_numpy(), internacionales["categoria"].to_numpy()
  for i in orden:
    r0, r1 = rating[home[i]], rating[away[i]]
    elo_0[i], elo_1[i] = r0, r1
    esperado = 1 / (1 + 10 ** (-(r0 - r1 + (0 if neutral[i] else VENTAJA_LOCAL)) / 400))
    resultado = 1.0 if g0[i] > g1[i] else 0.5 if g0[i] == g1[i] else 0.0
    cambio = K_POR_CATEGORIA.get(categoria[i], 30) * multiplicador_goles(g0[i] - g1[i]) * (resultado - esperado)
    rating[home[i]], rating[away[i]] = r0 + cambio, r1 - cambio
    despues_0[i], despues_1[i] = r0 + cambio, r1 - cambio
  return elo_0, elo_1, despues_0, despues_1


def confederacion_de_torneo(torneo):
  return next((c for prefijo, c in PREFIJOS_CONFEDERACION if torneo.startswith(prefijo)), None)


def confederaciones(internacionales, agno):
  """
  Confederacion de cada seleccion para el mundial (agno): la mas frecuente entre sus partidos de torneos continentales de los VENTANA_CONFEDERACION años previos (o de toda su historia si en esa ventana no jugo ninguno)
  """
  if "tournament" not in internacionales:
    return {}
  con_confed = internacionales.assign(confed=internacionales["tournament"].map(confederacion_de_torneo)).dropna(subset=["confed"])
  con_confed = con_confed[con_confed["agno"] <= agno]
  salida = {}
  for ventana in (con_confed[con_confed["agno"] > agno - VENTANA_CONFEDERACION], con_confed):
    conteo = defaultdict(Counter)
    for h, a, c in zip(ventana["home"], ventana["away"], ventana["confed"]):
      conteo[h][c] += 1
      conteo[a][c] += 1
    for pais, c in conteo.items():
      salida.setdefault(pais, c.most_common(1)[0][0])
  return salida


def contexto_de_mundiales(internacionales):
  """
  ¿QUE HACE?
  Para cada mundial de (internacionales): fecha de inicio, anfitriones, confederacion del anfitrion, Elo de cada seleccion al inicio y confederacion de cada seleccion

  Returns:

    - diccionario agno -> {"inicio", "anfitriones" (set), "confederacion_anfitrion" (set), "elo" (pais -> Elo al inicio), "confederacion" (pais -> confederacion)}
  """
  _, _, despues_0, despues_1 = calcular_elo(internacionales)
  mundiales = internacionales[internacionales["categoria"] == "mundial"]
  salida = {}
  for agno, partidos in mundiales.groupby("agno"):
    inicio = partidos["fecha"].min()
    anfitriones = set(partidos.loc[partidos["neutral"] == 0, "home"])
    # Elo al inicio: el rating de cada seleccion despues de su ultimo partido anterior a esa fecha
    previos = (internacionales["fecha"] < inicio).to_numpy()
    elo = {}
    for _, h, a, e0, e1 in sorted(zip(internacionales["fecha"].to_numpy()[previos], internacionales["home"].to_numpy()[previos],
                                      internacionales["away"].to_numpy()[previos], despues_0[previos], despues_1[previos])):
      elo[h], elo[a] = e0, e1
    confed = confederaciones(internacionales, agno)
    salida[int(agno)] = {"inicio": inicio, "anfitriones": anfitriones, "elo": elo, "confederacion": confed,
                         "confederacion_anfitrion": {confed.get(p) for p in anfitriones} - {None}}
  return salida


@lru_cache(maxsize=1)
def contexto_desde_bd():
  """"contexto_de_mundiales" de la tabla internacionales de la BD, calculado una vez por proceso (la app lo pide en cada consulta)"""
  from .modulo import formar_dataset_real
  return contexto_de_mundiales(formar_dataset_real("internacionales"))


def descansos(internacionales):
  """Diccionario (fecha, pais) -> dias desde el partido anterior de ese pais (hasta DESCANSO_MAXIMO), para cada partido de (internacionales)"""
  ordenados = internacionales.sort_values("fecha", kind="stable")
  ultimo, salida = {}, {}
  for fecha, h, a in zip(pd.to_datetime(ordenados["fecha"]), ordenados["home"], ordenados["away"]):
    for pais in (h, a):
      salida[(fecha.strftime("%Y-%m-%d"), pais)] = min((fecha - ultimo[pais]).days, DESCANSO_MAXIMO) if pais in ultimo else DESCANSO_MAXIMO
      ultimo[pais] = fecha
  return salida
