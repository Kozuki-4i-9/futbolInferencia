"""
Interfaz comun de predictores.

La simulacion solo necesita, por partido, los goles esperados de cada equipo (la lambda que usa la interseccion de
Poisson). Cualquier modelo que cumpla "Predictor" sirve, sin importar su framework: una LSTM de Keras, un modelo de
scikit-learn, un modelo fundacional de series temporales o un servicio externo.

Cada modelo entrenado vive en su carpeta de data/modelos/ con un "esquema.json" que dice que backend lo carga y con que
metricas se entreno. "cargar_predictor" se niega a usar un modelo cuyas metricas no coinciden con ALL_METRICS, que es
el error que dejo al .h5 de 2024 desalineado con el codigo sin que nada avisara.
"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol

import numpy as np
import pandas as pd

from .modulo import ALL_METRICS

ESQUEMA = "esquema.json"


@dataclass(frozen=True)
class Lote:
  """
  Entrada de un predictor, ya calculada por la simulacion o por "construir_dataset"

    - tabla: una fila por partido con home_id, away_id, ALL_METRICS_0 y ALL_METRICS_1 (metricas previas al partido)

    - secuencias: salida de "construir_secuencias" para esos mismos partidos (solo la usan los modelos de series temporales)
  """
  tabla: pd.DataFrame
  secuencias: Optional[dict] = None


# orden de columnas de la salida de "calculo_metricas_0" (COLS_MODELO_POR_EQUIPO por lado)
COLUMNAS_ENTRADA = [
  "home_id", *[f"{m}_0" for m in ALL_METRICS], "score_0",
  "away_id", *[f"{m}_1" for m in ALL_METRICS], "score_1",
]


def lote_desde_entrada(X):
  """
  ¿QUE HACE?
  Convierte el array (X) que arma la simulacion con "calculo_metricas_0" (una fila por partido) en el (Lote) que recibe cualquier predictor: los ids de los equipos y sus ALL_METRICS por nombre, sin los scores

  Returns:

    - Lote sin secuencias
  """
  tabla = pd.DataFrame(np.atleast_2d(X), columns=COLUMNAS_ENTRADA).drop(columns=["score_0", "score_1"])
  return Lote(tabla.astype(float).astype({"home_id": "int64", "away_id": "int64"}))


class Predictor(Protocol):
  nombre: str

  def predecir(self, lote: Lote) -> np.ndarray:
    """Goles esperados (lambda de Poisson) de home y away: array (n, 2), una fila por partido de (lote)"""
    ...

  def a_esquema(self) -> dict:
    """Todo lo necesario para reconstruir el predictor (se guarda en esquema.json); debe incluir "backend\""""
    ...


class PredictorMedias:
  """
  ¿QUE HACE?
  Linea base: predice para todo partido la media de goles de local y de visitante del entrenamiento (el "Poisson con medias" de la Fase 2). Cualquier modelo que no la supere no aporta

  Atributos:

    - media_0, media_1: goles medios de home y away
  """
  nombre = "medias"

  def __init__(self, media_0, media_1):
    self.media_0, self.media_1 = float(media_0), float(media_1)

  @classmethod
  def entrenar(cls, dataset):
    """Calcula las medias con las columnas goles_0 y goles_1 de (dataset) ("construir_dataset")"""
    return cls(dataset["goles_0"].mean(), dataset["goles_1"].mean())

  @classmethod
  def desde_esquema(cls, esquema, carpeta):
    return cls(esquema["media_0"], esquema["media_1"])

  def predecir(self, lote):
    return np.tile([self.media_0, self.media_1], (len(lote.tabla), 1))

  def a_esquema(self):
    return {"backend": self.nombre, "media_0": self.media_0, "media_1": self.media_1}


class PredictorMasPuntos:
  """
  ¿QUE HACE?
  Linea base (a) de la Fase 2: gana el equipo con mas puntos por partido en lo que va de su ciclo (eliminatorias y mundial hasta ese partido); predice 1-0 para el, y 1-1 si estan iguales. No se entrena
  """
  nombre = "mas_puntos"
  determinista = True

  @classmethod
  def desde_esquema(cls, esquema, carpeta):
    return cls()

  def predecir(self, lote):
    t = lote.tabla
    ppp_0 = (t["PTS_0"] / t["PJ_0"].where(t["PJ_0"] > 0)).fillna(0).to_numpy()
    ppp_1 = (t["PTS_1"] / t["PJ_1"].where(t["PJ_1"] > 0)).fillna(0).to_numpy()
    return np.stack([np.where(ppp_0 >= ppp_1, 1.0, 0.0), np.where(ppp_1 >= ppp_0, 1.0, 0.0)], axis=1)

  def a_esquema(self):
    return {"backend": self.nombre}


class PredictorEmpate:
  """Linea base (c) de la Fase 2: todo partido termina 1-1"""
  nombre = "empate"
  determinista = True

  @classmethod
  def desde_esquema(cls, esquema, carpeta):
    return cls()

  def predecir(self, lote):
    return np.ones((len(lote.tabla), 2))

  def a_esquema(self):
    return {"backend": self.nombre}


class PredictorKeras:
  """
  ¿QUE HACE?
  LSTM de series temporales: con la historia de cada equipo (sus ultimos k partidos, de "construir_secuencias") y su id, predice los goles esperados (lambda de Poisson) de los dos

  ¿COMO LO HACE?
  0-) Cada paso de la secuencia se normaliza con la media y desviacion del entrenamiento; los pasos de relleno quedan en 0 y una capa Masking los ignora

  1-) Un mismo LSTM resume la secuencia de cada equipo y una capa Embedding (entrenada con el modelo y guardada con el) representa su id

  2-) Una misma cabeza densa recibe (atacante, defensor) y devuelve con softplus los goles esperados del atacante: se aplica a (home, away) y a (away, home). La perdida es la de Poisson, coherente con la interseccion de Poisson de la simulacion

  Atributos:

    - modelo: keras.Model

    - config: k, con_rival, con_contexto, n_paises, media y desv por valor de cada paso, paises (nombre por id)
  """
  nombre = "lstm"
  ARCHIVO = "modelo.keras"

  def __init__(self, modelo, config):
    self.modelo, self.config = modelo, config

  @staticmethod
  def construir_modelo(n_pasos, n_valores, n_paises, unidades=32, dim_pais=8):
    """Arma y compila la red descrita en la clase; el id n_paises queda para equipos desconocidos"""
    import keras
    from keras import layers

    seq_0, seq_1 = keras.Input((n_pasos, n_valores), name="seq_0"), keras.Input((n_pasos, n_valores), name="seq_1")
    id_0, id_1 = keras.Input((1,), dtype="int32", name="id_0"), keras.Input((1,), dtype="int32", name="id_1")
    enmascarar, lstm = layers.Masking(0.0), layers.LSTM(unidades, name="historia")
    pais, aplanar = layers.Embedding(n_paises + 1, dim_pais, name="pais"), layers.Flatten()
    oculta, salida = layers.Dense(32, activation="relu", name="oculta"), layers.Dense(1, activation="softplus", name="goles")

    def equipo(seq, ids):
      return layers.Concatenate()([lstm(enmascarar(seq)), aplanar(pais(ids))])

    def goles(atacante, defensor):
      return salida(oculta(layers.Concatenate()([atacante, defensor])))

    e_0, e_1 = equipo(seq_0, id_0), equipo(seq_1, id_1)
    modelo = keras.Model([seq_0, seq_1, id_0, id_1], layers.Concatenate(name="lambdas")([goles(e_0, e_1), goles(e_1, e_0)]))
    modelo.compile(optimizer=keras.optimizers.Adam(1e-3), loss="poisson")
    return modelo

  def entradas(self, secuencias, filas=None):
    """Arrays de entrada a la red desde la salida de "construir_secuencias" (solo las (filas) indicadas)"""
    filas = slice(None) if filas is None else filas
    c = self.config
    salida = {}
    for i in (0, 1):
      seq = (secuencias[f"seq_{i}"][filas] - np.array(c["media"])) / np.array(c["desv"])
      salida[f"seq_{i}"] = np.where(secuencias[f"mascara_{i}"][filas][..., None] == 1, seq, 0.0).astype("float32")
      salida[f"id_{i}"] = np.minimum(secuencias[f"id_{i}"][filas], c["n_paises"]).reshape(-1, 1).astype("int32")
    return salida

  @classmethod
  def entrenar(cls, dataset, secuencias, filas_entrenamiento, filas_validacion, k, con_rival, con_contexto,
               semilla=0, epocas=80, paciencia=8, verbose=0):
    """
    ¿QUE HACE?
    Entrena la red con (filas_entrenamiento) y detiene el entrenamiento cuando la perdida en (filas_validacion) deja de mejorar (se quedan los mejores pesos). Sin (filas_validacion) entrena (epocas) fijas

    Args:

      - dataset: df de "construir_dataset" (para el nombre de cada id)

      - secuencias: salida de "construir_secuencias(dataset, k, con_rival, con_contexto)" con esos mismos parametros

      - filas_entrenamiento, filas_validacion: arrays booleanos o de indices sobre (dataset); (filas_validacion) puede ser None

      - semilla: fija la inicializacion y el orden de los lotes

    Returns:

      - (predictor, historia del entrenamiento)
    """
    import keras

    keras.utils.set_random_seed(semilla)
    pasos = np.concatenate([secuencias["seq_0"][filas_entrenamiento][secuencias["mascara_0"][filas_entrenamiento] == 1],
                            secuencias["seq_1"][filas_entrenamiento][secuencias["mascara_1"][filas_entrenamiento] == 1]])
    desv = pasos.std(axis=0)
    nombres = pd.concat([dataset[["home_id", "home"]].set_axis(["id", "pais"], axis=1),
                         dataset[["away_id", "away"]].set_axis(["id", "pais"], axis=1)]).drop_duplicates("id").set_index("id")["pais"]
    n_paises = int(nombres.index.max()) + 1
    config = {
      "k": k, "con_rival": con_rival, "con_contexto": con_contexto, "n_paises": n_paises, "semilla": semilla,
      "media": pasos.mean(axis=0).tolist(), "desv": np.where(desv > 0, desv, 1.0).tolist(),
      "paises": [nombres.get(i) for i in range(n_paises)],
    }
    predictor = cls(cls.construir_modelo(k, pasos.shape[1], n_paises), config)
    validar = filas_validacion is not None and np.any(filas_validacion)
    historia = predictor.modelo.fit(
      predictor.entradas(secuencias, filas_entrenamiento), secuencias["y"][filas_entrenamiento],
      validation_data=(predictor.entradas(secuencias, filas_validacion), secuencias["y"][filas_validacion]) if validar else None,
      epochs=epocas, batch_size=128, verbose=verbose,
      callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=paciencia, restore_best_weights=True)] if validar else [],
    )
    return predictor, historia.history

  @classmethod
  def desde_esquema(cls, esquema, carpeta):
    import keras
    return cls(keras.models.load_model(Path(carpeta) / cls.ARCHIVO), esquema["config"])

  def secuencias_desde_tabla(self, tabla):
    """
    ¿QUE HACE?
    Arma las secuencias de un paso (k = 1) para la simulacion de la app, con la misma "construir_secuencias" del entrenamiento

    ¿COMO LO HACE?
    0-) Si el modelo usa mas de un paso lanza ValueError: la simulacion no guarda la historia partido a partido de cada equipo

    1-) Traduce los equipos (nombres en "home" y "away") a los ids del entrenamiento (config "paises"); un equipo que no vio va al id de desconocidos

    2-) Llama a "construir_secuencias" con k = 1 y los bloques de la config; (tabla) debe traer ALL_METRICS_0/1 y, con contexto, neutral, categoria, pk_real_0/1 y sot_real_0/1
    """
    from .dataset import construir_secuencias

    c = self.config
    if c["k"] != 1:
      raise ValueError(f"La simulacion solo arma secuencias de 1 paso y este modelo usa k={c['k']}")
    ids = {nombre: i for i, nombre in enumerate(c["paises"]) if nombre is not None}
    tabla = tabla.assign(
      home_id=tabla["home"].map(ids).fillna(c["n_paises"]).astype("int64"),
      away_id=tabla["away"].map(ids).fillna(c["n_paises"]).astype("int64"),
      goles_0=0, goles_1=0,
    )
    return construir_secuencias(tabla, k=1, con_rival=c["con_rival"], con_contexto=c["con_contexto"])

  def predecir(self, lote):
    secuencias = lote.secuencias if lote.secuencias is not None else self.secuencias_desde_tabla(lote.tabla)
    return self.modelo.predict(self.entradas(secuencias), verbose=0, batch_size=512)

  def guardar_pesos(self, carpeta):
    Path(carpeta).mkdir(parents=True, exist_ok=True)
    self.modelo.save(Path(carpeta) / self.ARCHIVO)

  def a_esquema(self):
    return {"backend": self.nombre, "config": self.config}


# backend -> constructor desde (esquema, carpeta); cada backend nuevo se registra aqui
REGISTRO = {
  PredictorMedias.nombre: PredictorMedias.desde_esquema,
  PredictorMasPuntos.nombre: PredictorMasPuntos.desde_esquema,
  PredictorEmpate.nombre: PredictorEmpate.desde_esquema,
  PredictorKeras.nombre: PredictorKeras.desde_esquema,
}


def guardar_predictor(predictor, carpeta, **extra):
  """
  ¿QUE HACE?
  Escribe en (carpeta)/esquema.json lo que reconstruye a (predictor), las metricas con que se entreno (ALL_METRICS en su orden) y lo que se pase en (extra) (p. ej. evaluacion, pais_id, n_steps). Los pesos, si los hay, los guarda el propio backend en la misma carpeta
  """
  carpeta = Path(carpeta)
  carpeta.mkdir(parents=True, exist_ok=True)
  if hasattr(predictor, "guardar_pesos"):
    predictor.guardar_pesos(carpeta)
  esquema = {**predictor.a_esquema(), "metricas": list(ALL_METRICS), **extra}
  (carpeta / ESQUEMA).write_text(json.dumps(esquema, indent=2, ensure_ascii=False), encoding="utf-8")


def cargar_predictor(carpeta):
  """
  ¿QUE HACE?
  Reconstruye el predictor de (carpeta) con el backend que indica su esquema.json

  ¿COMO LO HACE?
  0-) Lee el esquema; si sus metricas no son exactamente ALL_METRICS (mismas y en el mismo orden) lanza ValueError: ese modelo se entreno con otra entrada y predeciria mal

  1-) Busca el backend en (REGISTRO); si no esta registrado lanza ValueError

  Returns:

    - (predictor, esquema)
  """
  carpeta = Path(carpeta)
  esquema = json.loads((carpeta / ESQUEMA).read_text(encoding="utf-8"))
  if esquema.get("metricas") != list(ALL_METRICS):
    raise ValueError(f"{carpeta.name} se entreno con las metricas {esquema.get('metricas')} y el codigo usa {list(ALL_METRICS)}; hay que reentrenarlo")
  if esquema.get("backend") not in REGISTRO:
    raise ValueError(f"Backend desconocido {esquema.get('backend')!r}; registrados: {sorted(REGISTRO)}")
  return REGISTRO[esquema["backend"]](esquema, carpeta), esquema
