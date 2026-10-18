"""
Constructor del dataset de entrenamiento.

Genera, con el mismo codigo que usa la app para simular (calculo_metricas_0 y calculo_metricas_1), una fila por partido
real con las metricas de los dos equipos ANTES del partido y los goles que hicieron. Encima de esa tabla arma las
secuencias por equipo (sus ultimos K partidos) que consume un modelo de series temporales.

Ningun partido ve su propio resultado ni datos de mundiales posteriores:
  - las metricas base se acumulan partido a partido y cada fila guarda el estado previo al partido
  - el historico de metricas abstractas de un mundial solo suma mundiales anteriores ("historico_abstractas")
"""
import numpy as np
import pandas as pd

from . import modulo
from .modulo import BASE_METRICS, ALL_METRICS, calculo_metricas_0, calculo_metricas_1, historico_abstractas

FASE_ELIMINATORIA = "eliminatoria"
FASE_MUNDIAL = "mundial"


def mapa_paises(*dfs):
  """
  ¿QUE HACE?
  Asigna a cada pais un id estable: el orden alfabetico de todos los nombres que aparecen como home o away en (dfs). El mismo mapa debe guardarse junto al modelo entrenado, para que el id de un pais signifique lo mismo al predecir

  Returns:

    - diccionario nombre del pais -> id
  """
  nombres = sorted(set().union(*(set(df["home"]) | set(df["away"]) for df in dfs)))
  return {nombre: i for i, nombre in enumerate(nombres)}


def separar_score(df):
  """Pasa la columna "score" ("3:2") de la tabla partidos a enteros score_0 y score_1"""
  goles = df["score"].str.split(":", expand=True).astype("int64")
  return df.drop(columns="score").assign(score_0=goles[0], score_1=goles[1])


def cargar_fuentes():
  """
  ¿QUE HACE?
  Lee de la BD las tres tablas que necesita "construir_dataset": partidos de mundiales, partidos de eliminatorias y metricas abstractas por pais y mundial

  Returns:

    - (partidos, clasificaciones, proxy): df con home, away, score_0, score_1, agno (los dos primeros) y df de "formar_dataset_real('proxy_desempegno')"
  """
  partidos = separar_score(modulo.formar_dataset_real("partidos"))
  clasificaciones = modulo.formar_dataset_real("clasificaciones")
  proxy = modulo.formar_dataset_real("proxy_desempegno")
  return partidos, clasificaciones, proxy


def _jugar(estado, home, away, goles_0, goles_1):
  """
  ¿QUE HACE?
  Procesa un partido real: devuelve las metricas de entrada al modelo de ambos equipos antes del partido y deja en (estado) sus metricas base despues del partido

  ¿COMO LO HACE?
  0-) Arma un df de fase de una fila con las metricas base previas de cada equipo (0 si es su primer partido del ciclo) y el resultado real

  1-) "calculo_metricas_0" deriva las ALL_METRICS de ambos lados con el estado previo (y el progreso abstracto en "modulo.opcionales")

  2-) "calculo_metricas_1" acumula el resultado: la fila pasa a tener el estado posterior y "modulo.opcionales" avanza el progreso abstracto

  3-) Guarda en (estado) el estado posterior de ambos equipos

  Args:

    - estado: diccionario id -> {metrica base: valor}; se actualiza in-place

    - home, away: ids de los equipos

    - goles_0, goles_1: goles reales de home y away

  Returns:

    - Serie con ALL_METRICS_0 y ALL_METRICS_1 previas al partido
  """
  ceros = {m: 0.0 for m in BASE_METRICS}
  previo_0, previo_1 = estado.get(home, ceros), estado.get(away, ceros)
  fila = pd.DataFrame([{
    "home": home, **{f"{m}_0": float(previo_0[m]) for m in BASE_METRICS}, "score_0": goles_0,
    "away": away, **{f"{m}_1": float(previo_1[m]) for m in BASE_METRICS}, "score_1": goles_1,
  }])

  entrada = calculo_metricas_0(fila).iloc[0]
  calculo_metricas_1(fila)

  estado[home] = {m: fila.at[0, f"{m}_0"] for m in BASE_METRICS}
  estado[away] = {m: fila.at[0, f"{m}_1"] for m in BASE_METRICS}
  return entrada[[f"{m}_{i}" for i in (0, 1) for m in ALL_METRICS]]


def construir_dataset(partidos, clasificaciones, proxy, pais_id=None, agnos=None):
  """
  ¿QUE HACE?
  Arma la tabla de entrenamiento: una fila por partido real (eliminatorias y mundiales) en orden cronologico, con las metricas de los dos equipos antes del partido y sus goles. Reproduce el ciclo que simula la app: cada mundial arranca con lo acumulado en sus eliminatorias

  ¿COMO LO HACE?
  0-) Si no se pasa (pais_id), lo arma con "mapa_paises"

  1-) Para cada mundial Y de (agnos), en orden:

    1.0) Eliminatorias del ciclo (agno en (mundial anterior, Y]): las metricas base arrancan en 0 para todos (como "obtener_dic_agnos") y el progreso abstracto desde "historico_abstractas" hasta Y. Cada partido se procesa con "_jugar"

    1.1) Se guarda el estado al cerrar las eliminatorias del ciclo

    1.2) Mundial Y: el progreso abstracto vuelve al historico previo a Y (como "reiniciar_opcionales" en la app). Un equipo sin eliminatorias en este ciclo (anfitrion, campeon vigente) arranca con su estado de eliminatorias del ciclo mas reciente en que jugo, o en 0 (como "agregar_features")

  2-) Restaura "modulo.opcionales" al terminar, para no dejar estado global de entrenamiento en el proceso

  Args:

    - partidos: df de mundiales con home, away, score_0, score_1 y agno (en orden de juego dentro de cada año)

    - clasificaciones: df de eliminatorias con home, away, score_0, score_1 y agno (en orden de juego)

    - proxy: df con pais, agno y METRICAS_ABSTRACTAS (puede estar vacio)

    - pais_id: diccionario nombre del país -> id; None lo arma con todos los equipos de (partidos) y (clasificaciones)

    - agnos: mundiales a procesar; None usa "modulo.agnos" (1934 - 2022)

  Returns:

    - df con columnas agno, fase, orden, home, away, home_id, away_id, ALL_METRICS_0, ALL_METRICS_1, goles_0, goles_1
  """
  pais_id = pais_id or mapa_paises(partidos, clasificaciones)
  agnos = sorted(agnos or modulo.agnos)
  opcionales_previos = modulo.opcionales
  filas, cierre_eliminatorias = [], {}

  def procesar(df_partidos, agno, fase, estado):
    for orden, p in enumerate(df_partidos.itertuples(index=False)):
      home, away = pais_id[p.home], pais_id[p.away]
      entrada = _jugar(estado, home, away, int(p.score_0), int(p.score_1))
      filas.append({
        "agno": agno, "fase": fase, "orden": orden, "home": p.home, "away": p.away, "home_id": home, "away_id": away,
        **entrada.to_dict(), "goles_0": int(p.score_0), "goles_1": int(p.score_1),
      })

  try:
    for i, agno in enumerate(agnos):
      desde = agnos[i - 1] if i > 0 else -np.inf

      # 1.0) eliminatorias del ciclo
      modulo.opcionales = historico_abstractas(proxy, pais_id, agno)
      estado = {}
      eliminatorias = clasificaciones[(clasificaciones["agno"] > desde) & (clasificaciones["agno"] <= agno)]
      procesar(eliminatorias, agno, FASE_ELIMINATORIA, estado)

      # 1.1) estado al cerrar las eliminatorias
      cierre_eliminatorias[agno] = dict(estado)

      # 1.2) mundial
      modulo.opcionales = historico_abstractas(proxy, pais_id, agno)
      mundial = partidos[partidos["agno"] == agno]
      estado_mundial = dict(estado)
      for nombre in set(mundial["home"]) | set(mundial["away"]):
        equipo = pais_id[nombre]
        if equipo not in estado_mundial:
          previos = [cierre_eliminatorias[a][equipo] for a in reversed(agnos[:i]) if equipo in cierre_eliminatorias[a]]
          if previos:
            estado_mundial[equipo] = previos[0]
      procesar(mundial, agno, FASE_MUNDIAL, estado_mundial)
  finally:
    # 2-) sin estado global residual
    modulo.opcionales = opcionales_previos

  return pd.DataFrame(filas)


def construir_secuencias(dataset, k=5, con_rival=True):
  """
  ¿QUE HACE?
  Convierte la tabla de "construir_dataset" en muestras de serie temporal: para cada partido, la historia de cada equipo como la secuencia de sus ultimos (k) partidos, terminando en el partido a predecir. Cada paso lleva las ALL_METRICS previas del equipo y, si (con_rival), tambien las de su rival en ese partido: los goles que hace un equipo dependen de a quien enfrenta

  ¿COMO LO HACE?
  0-) Recorre la tabla en orden y guarda, por equipo, la lista de sus pasos: sus metricas (las de su lado: _0 si jugo de local, _1 si de visitante) seguidas de las del otro lado

  1-) Para cada partido, toma de cada equipo sus ultimos (k) pasos incluido el actual (que solo contiene informacion previa al partido); si tiene menos de (k) rellena al principio con ceros y lo marca en la mascara

  Args:

    - dataset: df de "construir_dataset" (orden cronologico)

    - k: largo de la secuencia (N_STEPS del modelo)

    - con_rival: True agrega a cada paso las ALL_METRICS del rival (2 * len(ALL_METRICS) por paso)

  Returns:

    - diccionario con arrays: "id_0", "id_1" (n,), "seq_0", "seq_1" (n, k, len(ALL_METRICS) o el doble con rival), "mascara_0", "mascara_1" (n, k) con 1 en pasos reales, "y" (n, 2) goles de home y away
  """
  n, f = len(dataset), len(ALL_METRICS) * (2 if con_rival else 1)
  salida = {
    "id_0": dataset["home_id"].to_numpy(), "id_1": dataset["away_id"].to_numpy(),
    "seq_0": np.zeros((n, k, f)), "seq_1": np.zeros((n, k, f)),
    "mascara_0": np.zeros((n, k)), "mascara_1": np.zeros((n, k)),
    "y": dataset[["goles_0", "goles_1"]].to_numpy(dtype=float),
  }
  historia = {}
  cols = {i: [f"{m}_{i}" for m in ALL_METRICS] for i in (0, 1)}
  matriz = {i: dataset[cols[i]].to_numpy(dtype=float) for i in (0, 1)}

  for fila, (id_0, id_1) in enumerate(zip(salida["id_0"], salida["id_1"])):
    for lado, equipo in ((0, id_0), (1, id_1)):
      pasos = historia.setdefault(equipo, [])
      paso = matriz[lado][fila]
      if con_rival:
        paso = np.concatenate([paso, matriz[1 - lado][fila]])
      pasos.append(paso)
      ultimos = pasos[-k:]
      salida[f"seq_{lado}"][fila, k - len(ultimos):] = ultimos
      salida[f"mascara_{lado}"][fila, k - len(ultimos):] = 1

  return salida
