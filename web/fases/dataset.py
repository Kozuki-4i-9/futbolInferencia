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

# ciclo previo a cada mundial: solo sus eliminatorias (lo que usa la app hoy) o todos los partidos de selecciones
CICLO_ELIMINATORIAS = "eliminatorias"
CICLO_INTERNACIONALES = "internacionales"

# tipo de partido; mismo orden que etl/busca_internacionales.py (el modelo la recibe como one-hot)
CATEGORIAS = ["mundial", "eliminatoria_mundial", "continental", "eliminatoria_continental", "liga_naciones", "amistoso", "regional"]


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
  Lee de la BD las tablas que necesita "construir_dataset": partidos de mundiales, partidos de eliminatorias, metricas abstractas por pais y mundial, y todos los partidos internacionales

  Returns:

    - (partidos, clasificaciones, proxy, internacionales): df con home, away, score_0, score_1, agno (los dos primeros), df de "formar_dataset_real('proxy_desempegno')" y df de "formar_dataset_real('internacionales')" (vacio si no se cargo)
  """
  partidos = modulo.formar_dataset_real("partidos")
  # el marcador es el de los 90 o 120 minutos (la tanda queda aparte); los partidos no jugados (jugado = 0) no entran
  partidos = separar_score(partidos[partidos["jugado"] == 1].drop(columns=["penales", "jugado"]))
  clasificaciones = modulo.formar_dataset_real("clasificaciones")
  proxy = modulo.formar_dataset_real("proxy_desempegno")
  internacionales = modulo.formar_dataset_real("internacionales")
  return partidos, clasificaciones, proxy, internacionales


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


def construir_dataset(partidos, clasificaciones, proxy, pais_id=None, agnos=None, internacionales=None, ciclo=CICLO_ELIMINATORIAS):
  """
  ¿QUE HACE?
  Arma la tabla de entrenamiento: una fila por partido real (ciclo previo y mundial) en orden cronologico, con las metricas de los dos equipos antes del partido, sus goles, el tipo de partido y si se jugo en campo neutral. Reproduce el ciclo que simula la app: cada mundial arranca con lo acumulado en su ciclo previo

  ¿COMO LO HACE?
  0-) Si no se pasa (pais_id), lo arma con "mapa_paises" (incluye los equipos de (internacionales) si se usan para el ciclo)

  1-) Para cada mundial Y de (agnos), en orden:

    1.0) Ciclo previo: las metricas base arrancan en 0 para todos (como "obtener_dic_agnos") y el progreso abstracto desde "historico_abstractas" hasta Y. Cada partido se procesa con "_jugar"

      - (ciclo) "eliminatorias": las eliminatorias con agno en (mundial anterior, Y]

      - (ciclo) "internacionales": todos los partidos de (internacionales) que no son de mundial, con fecha desde el inicio del mundial anterior hasta antes del inicio de Y, en orden de fecha

    1.1) Se guarda el estado al cerrar el ciclo

    1.2) Mundial Y: el progreso abstracto vuelve al historico previo a Y (como "reiniciar_opcionales" en la app). Un equipo sin partidos en este ciclo (anfitrion, campeon vigente) arranca con su estado al cierre del ciclo mas reciente en que jugo, o en 0 (como "agregar_features"). (neutral) sale de (internacionales); si no esta, vale 1

  2-) Restaura "modulo.opcionales" al terminar, para no dejar estado global de entrenamiento en el proceso

  Args:

    - partidos: df de mundiales con home, away, score_0, score_1 y agno (en orden de juego dentro de cada año)

    - clasificaciones: df de eliminatorias con home, away, score_0, score_1 y agno (en orden de juego)

    - proxy: df con pais, agno y METRICAS_ABSTRACTAS (puede estar vacio)

    - pais_id: diccionario nombre del país -> id; None lo arma con todos los equipos de las fuentes usadas

    - agnos: mundiales a procesar; None usa "modulo.agnos" (1934 - 2022)

    - internacionales: df de "formar_dataset_real('internacionales')" (fecha, agno, home, away, score_0, score_1, categoria, neutral) o None

    - ciclo: CICLO_ELIMINATORIAS o CICLO_INTERNACIONALES (este exige (internacionales))

  Returns:

    - df con columnas agno, fase, orden, home, away, home_id, away_id, categoria, neutral, ALL_METRICS_0, ALL_METRICS_1, goles_0, goles_1
  """
  hay_internacionales = internacionales is not None and len(internacionales) > 0
  if ciclo == CICLO_INTERNACIONALES and not hay_internacionales:
    raise ValueError("el ciclo 'internacionales' necesita la tabla internacionales (etl/busca_internacionales.py y load_data)")
  fuentes = [partidos, clasificaciones] + ([internacionales] if ciclo == CICLO_INTERNACIONALES else [])
  pais_id = pais_id or mapa_paises(*fuentes)
  agnos = sorted(agnos or modulo.agnos)
  opcionales_previos = modulo.opcionales
  filas, cierre_ciclo = [], {}

  if hay_internacionales:
    en_mundial = internacionales["categoria"] == "mundial"
    inicio_mundial = internacionales[en_mundial].groupby("agno")["fecha"].min().to_dict()
    neutral_mundial = {(r.agno, r.home, r.away): r.neutral for r in internacionales[en_mundial].itertuples()}
    resto = internacionales[~en_mundial]
  else:
    inicio_mundial, neutral_mundial = {}, {}

  def neutral_de(agno, home, away):
    return neutral_mundial.get((agno, home, away), neutral_mundial.get((agno, away, home), 1))

  def procesar(df_partidos, agno, fase, estado):
    for orden, p in enumerate(df_partidos.itertuples(index=False)):
      home, away = pais_id[p.home], pais_id[p.away]
      entrada = _jugar(estado, home, away, int(p.score_0), int(p.score_1))
      filas.append({
        "agno": agno, "fase": fase, "orden": orden, "home": p.home, "away": p.away, "home_id": home, "away_id": away,
        "categoria": p.categoria, "neutral": int(p.neutral),
        **entrada.to_dict(), "goles_0": int(p.score_0), "goles_1": int(p.score_1),
      })

  try:
    for i, agno in enumerate(agnos):
      desde = agnos[i - 1] if i > 0 else -np.inf

      # 1.0) ciclo previo
      modulo.opcionales = historico_abstractas(proxy, pais_id, agno)
      estado = {}
      if ciclo == CICLO_INTERNACIONALES:
        fecha_desde = inicio_mundial[desde] if i > 0 else ""
        previos = resto[(resto["fecha"] >= fecha_desde) & (resto["fecha"] < inicio_mundial[agno])]
      else:
        previos = clasificaciones[(clasificaciones["agno"] > desde) & (clasificaciones["agno"] <= agno)]
        previos = previos.assign(categoria="eliminatoria_mundial", neutral=0)
      procesar(previos, agno, FASE_ELIMINATORIA, estado)

      # 1.1) estado al cerrar el ciclo
      cierre_ciclo[agno] = dict(estado)

      # 1.2) mundial
      modulo.opcionales = historico_abstractas(proxy, pais_id, agno)
      mundial = partidos[partidos["agno"] == agno]
      mundial = mundial.assign(categoria="mundial", neutral=[neutral_de(agno, h, a) for h, a in zip(mundial["home"], mundial["away"])])
      estado_mundial = dict(estado)
      for nombre in set(mundial["home"]) | set(mundial["away"]):
        equipo = pais_id[nombre]
        if equipo not in estado_mundial:
          anteriores = [cierre_ciclo[a][equipo] for a in reversed(agnos[:i]) if equipo in cierre_ciclo[a]]
          if anteriores:
            estado_mundial[equipo] = anteriores[0]
      procesar(mundial, agno, FASE_MUNDIAL, estado_mundial)
  finally:
    # 2-) sin estado global residual
    modulo.opcionales = opcionales_previos

  return pd.DataFrame(filas)


def construir_secuencias(dataset, k=5, con_rival=True, con_contexto=True):
  """
  ¿QUE HACE?
  Convierte la tabla de "construir_dataset" en muestras de serie temporal: para cada partido, la historia de cada equipo como la secuencia de sus ultimos (k) partidos, terminando en el partido a predecir. Cada paso lleva las ALL_METRICS previas del equipo y, si (con_rival), tambien las de su rival en ese partido: los goles que hace un equipo dependen de a quien enfrenta. Si (con_contexto), cada paso suma el contexto del partido: si el equipo jugo de local en su cancha (local y no neutral), si fue en campo neutral y su categoria en one-hot (CATEGORIAS)

  ¿COMO LO HACE?
  0-) Recorre la tabla en orden y guarda, por equipo, la lista de sus pasos: sus metricas (las de su lado: _0 si jugo de local, _1 si de visitante) seguidas de las del otro lado

  1-) Para cada partido, toma de cada equipo sus ultimos (k) pasos incluido el actual (que solo contiene informacion previa al partido); si tiene menos de (k) rellena al principio con ceros y lo marca en la mascara

  Args:

    - dataset: df de "construir_dataset" (orden cronologico)

    - k: largo de la secuencia (N_STEPS del modelo)

    - con_rival: True agrega a cada paso las ALL_METRICS del rival (2 * len(ALL_METRICS) por paso)

    - con_contexto: True agrega a cada paso 2 + len(CATEGORIAS) valores de contexto (es_local, neutral, one-hot de categoria)

  Returns:

    - diccionario con arrays: "id_0", "id_1" (n,), "seq_0", "seq_1" (n, k, len(ALL_METRICS), el doble con rival, mas 2 + len(CATEGORIAS) con contexto), "mascara_0", "mascara_1" (n, k) con 1 en pasos reales, "y" (n, 2) goles de home y away
  """
  n, f = len(dataset), len(ALL_METRICS) * (2 if con_rival else 1) + (2 + len(CATEGORIAS) if con_contexto else 0)
  salida = {
    "id_0": dataset["home_id"].to_numpy(), "id_1": dataset["away_id"].to_numpy(),
    "seq_0": np.zeros((n, k, f)), "seq_1": np.zeros((n, k, f)),
    "mascara_0": np.zeros((n, k)), "mascara_1": np.zeros((n, k)),
    "y": dataset[["goles_0", "goles_1"]].to_numpy(dtype=float),
  }
  historia = {}
  cols = {i: [f"{m}_{i}" for m in ALL_METRICS] for i in (0, 1)}
  matriz = {i: dataset[cols[i]].to_numpy(dtype=float) for i in (0, 1)}
  if con_contexto:
    neutral = dataset["neutral"].to_numpy(dtype=float)
    categoria = np.array([[c == cat for cat in CATEGORIAS] for c in dataset["categoria"]], dtype=float)

  for fila, (id_0, id_1) in enumerate(zip(salida["id_0"], salida["id_1"])):
    for lado, equipo in ((0, id_0), (1, id_1)):
      pasos = historia.setdefault(equipo, [])
      paso = matriz[lado][fila]
      if con_rival:
        paso = np.concatenate([paso, matriz[1 - lado][fila]])
      if con_contexto:
        es_local = float(lado == 0 and neutral[fila] == 0)
        paso = np.concatenate([paso, [es_local, neutral[fila]], categoria[fila]])
      pasos.append(paso)
      ultimos = pasos[-k:]
      salida[f"seq_{lado}"][fila, k - len(ultimos):] = ultimos
      salida[f"mascara_{lado}"][fila, k - len(ultimos):] = 1

  return salida
