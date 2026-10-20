"""
Evaluacion de predictores (Fase 2): las metricas de la guia sobre una particion temporal del dataset.

Particion (sin mirar el futuro):
  - entrenamiento: todo lo jugado en los ciclos hasta (hasta) inclusive (eliminatorias y mundiales)
  - validacion: el ciclo siguiente completo (sus eliminatorias y su mundial), para elegir configuracion
  - prueba: los partidos de mundial de los años de prueba (2018 y 2022), que solo se miran al final

Metricas, con los goles esperados (lambda) que devuelve cada predictor:
  - mae_goles: error absoluto medio entre lambda y los goles reales (de los dos equipos)
  - acierto_resultado: % de partidos donde el resultado mas probable (local / empate / visitante, con dos Poisson
    independientes) coincide con el real
  - marcador_exacto: % de partidos donde el marcador mas probable de cada equipo (la moda de su Poisson) es el real
  - deviance: perdida de Poisson media (la que minimiza el LSTM; menor es mejor)

Las lineas base deterministas (siempre 1-1, gana el de mas puntos) declaran su marcador con el atributo
"determinista": su resultado y su marcador son los que dicen, no los de pasar sus numeros por Poisson.
"""
import numpy as np
from scipy.stats import poisson

from .dataset import FASE_MUNDIAL
from .predictores import Lote

MAX_GOLES = 10


def probabilidades_resultado(lambdas):
  """(n, 2) goles esperados -> (n, 3) probabilidades de gana local, empate y gana visitante con dos Poisson independientes"""
  goles = np.arange(MAX_GOLES + 1)
  p_0 = poisson.pmf(goles[None, :], np.maximum(lambdas[:, [0]], 1e-9))
  p_1 = poisson.pmf(goles[None, :], np.maximum(lambdas[:, [1]], 1e-9))
  conjunta = p_0[:, :, None] * p_1[:, None, :]
  return np.stack([np.tril(conjunta, -1).sum(axis=(1, 2)), np.trace(conjunta, axis1=1, axis2=2),
                   np.triu(conjunta, 1).sum(axis=(1, 2))], axis=1)


def metricas(goles, lambdas, determinista=False):
  """
  ¿QUE HACE?
  Calcula las metricas de la guia para (lambdas) (n, 2) contra los (goles) reales (n, 2); si (determinista), (lambdas) es directamente el marcador predicho

  Returns:

    - diccionario con partidos, mae_goles, acierto_resultado, marcador_exacto (en %) y deviance
  """
  goles, lambdas = np.asarray(goles, dtype=float), np.asarray(lambdas, dtype=float)
  real = np.sign(goles[:, 0] - goles[:, 1])                        # 1 local, 0 empate, -1 visitante
  if determinista:
    predicho, moda = np.sign(lambdas[:, 0] - lambdas[:, 1]), np.round(lambdas)
  else:
    predicho = 1 - probabilidades_resultado(lambdas).argmax(axis=1)  # 0 -> 1, 1 -> 0, 2 -> -1
    moda = np.floor(lambdas)
  l = np.maximum(lambdas, 1e-9)
  return {
    "partidos": int(len(goles)),
    "mae_goles": float(np.abs(lambdas - goles).mean()),
    "acierto_resultado": float((predicho == real).mean() * 100),
    "marcador_exacto": float((moda == goles).all(axis=1).mean() * 100),
    "deviance": float((l - goles * np.log(l)).mean()),
  }


def particion(dataset, hasta, validacion=None, prueba=None):
  """
  ¿QUE HACE?
  Filas booleanas de entrenamiento, validacion y prueba sobre (dataset) segun la particion del modulo

  Args:

    - hasta: ultimo mundial (ciclo) de entrenamiento

    - validacion: mundial cuyo ciclo completo valida (None: ninguno)

    - prueba: mundiales cuyos partidos de mundial se usan de prueba (None: ninguno)

  Returns:

    - (entrenamiento, validacion, prueba): arrays booleanos
  """
  agno = dataset["agno"].to_numpy()
  vacio = np.zeros(len(dataset), dtype=bool)
  return (
    agno <= hasta,
    agno == validacion if validacion is not None else vacio,
    np.isin(agno, prueba) & (dataset["fase"].to_numpy() == FASE_MUNDIAL) if prueba is not None else vacio,
  )


def lote(dataset, secuencias, filas):
  """Lote de (filas) de (dataset) con sus secuencias, para cualquier predictor"""
  return Lote(dataset[filas].reset_index(drop=True), {c: v[filas] for c, v in secuencias.items()} if secuencias else None)


def evaluar(predictor, dataset, secuencias, filas):
  """Metricas de (predictor) en las (filas) de (dataset)"""
  return metricas(dataset.loc[filas, ["goles_0", "goles_1"]].to_numpy(), predictor.predecir(lote(dataset, secuencias, filas)),
                  determinista=getattr(predictor, "determinista", False))
