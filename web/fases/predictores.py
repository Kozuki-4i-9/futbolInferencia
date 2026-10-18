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


# backend -> constructor desde (esquema, carpeta); cada backend nuevo se registra aqui
REGISTRO = {
  PredictorMedias.nombre: PredictorMedias.desde_esquema,
}


def guardar_predictor(predictor, carpeta, **extra):
  """
  ¿QUE HACE?
  Escribe en (carpeta)/esquema.json lo que reconstruye a (predictor), las metricas con que se entreno (ALL_METRICS en su orden) y lo que se pase en (extra) (p. ej. evaluacion, pais_id, n_steps). Los pesos, si los hay, los guarda el propio backend en la misma carpeta
  """
  carpeta = Path(carpeta)
  carpeta.mkdir(parents=True, exist_ok=True)
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
