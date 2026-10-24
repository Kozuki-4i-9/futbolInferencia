import json
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from fases.dataset import construir_secuencias
from fases.evaluacion import evaluar, particion
from fases.predictores import PredictorEmpate, PredictorKeras, PredictorMasPuntos, PredictorMedias, guardar_predictor

RAIZ = Path(settings.BASE_DIR).parent
DATASET = RAIZ / "data" / "datasets" / "partidos_features.csv"
RESULTADOS = RAIZ / "experimentos" / "resultados_baseline.json"

KS = [1, 3, 5, 8, 12]
SEMILLAS = [0, 1, 2]
METRICAS = ["mae_goles", "acierto_resultado", "marcador_exacto", "deviance"]


def resumen(corridas):
  """Media y desviacion de cada metrica sobre varias semillas"""
  return {m: {"media": float(np.mean([c[m] for c in corridas])), "desv": float(np.std([c[m] for c in corridas]))} for m in METRICAS}


class Command(BaseCommand):
  """
  ¿QUE HACE?
  "python manage.py evaluar_modelos" es la Fase 2 completa: elige la configuracion del LSTM con validacion temporal, lo mide una vez en los mundiales 2018 y 2022 contra las tres lineas base y guarda todo en experimentos/resultados_baseline.json (y el modelo en data/modelos/lstm-<fecha>/)

  ¿COMO LO HACE?
  0-) Seleccion: entrena con los ciclos hasta 2010 y valida con el ciclo 2014 (eliminatorias y mundial), 3 semillas por configuracion

    0.0) Largo de la secuencia K en KS (con rival y con contexto); gana el K mas chico cuya deviance media este a menos de una desviacion de la mejor

    0.1) Con ese K, se prueba quitar las metricas del rival, el contexto o ambos; se queda la variante mas simple que no empeora mas de una desviacion

  1-) Prueba: reentrena la configuracion elegida con los ciclos hasta 2014 (epocas fijas = media de las mejores epocas de la seleccion) y la mide en los partidos de mundial de 2018 y 2022, igual que las lineas base (medias entrenadas hasta 2014, mas puntos y siempre 1-1)

  2-) Guarda el modelo de la semilla 0 y el json de resultados

  Tarda unos 10 minutos en una CPU de 4 nucleos; imprime el avance de cada entrenamiento
  """
  help = "Fase 2: seleccion del LSTM, prueba contra lineas base y resultados en experimentos/resultados_baseline.json"
  requires_system_checks = []

  def avisar(self, texto):
    self.stdout.write(texto)
    self.stdout.flush()

  def entrenar_varias(self, ds, k, con_rival, con_contexto, filas_tr, filas_va, filas_ev, epocas=None):
    """Entrena una configuracion con cada semilla; devuelve (metricas por semilla, mejores epocas, predictores)"""
    sec = construir_secuencias(ds, k=k, con_rival=con_rival, con_contexto=con_contexto)
    corridas, epocas_mejores, predictores = [], [], []
    for semilla in SEMILLAS:
      inicio = time.time()
      if epocas is None:
        predictor, historia = PredictorKeras.entrenar(ds, sec, filas_tr, filas_va, k, con_rival, con_contexto, semilla=semilla)
        epocas_mejores.append(int(np.argmin(historia["val_loss"])) + 1)
      else:
        predictor, _ = PredictorKeras.entrenar(ds, sec, filas_tr, None, k, con_rival, con_contexto, semilla=semilla, epocas=epocas)
      resultado = evaluar(predictor, ds, sec, filas_ev)
      corridas.append(resultado)
      predictores.append(predictor)
      self.avisar(f"  k={k:<2} rival={con_rival!s:<5} contexto={con_contexto!s:<5} semilla={semilla}: "
                  f"deviance {resultado['deviance']:.4f}, acierto {resultado['acierto_resultado']:.1f}% ({time.time() - inicio:.0f}s)")
    return corridas, epocas_mejores, predictores

  def handle(self, *args, **opciones):
    if not DATASET.exists():
      raise CommandError(f"No existe {DATASET}: ejecutar antes 'python manage.py construir_dataset'")
    ds = pd.read_csv(DATASET)
    tr, va, _ = particion(ds, hasta=2010, validacion=2014)
    tr_final, _, prueba = particion(ds, hasta=2014, prueba=[2018, 2022])
    self.avisar(f"Dataset: {len(ds)} partidos; seleccion {tr.sum()} entrenamiento / {va.sum()} validacion; prueba {prueba.sum()} partidos de mundial")

    # 0.0) largo de la secuencia
    self.avisar("Seleccion de K (con rival y contexto)")
    seleccion = []
    for k in KS:
      corridas, epocas, _ = self.entrenar_varias(ds, k, True, True, tr, va, va)
      seleccion.append({"k": k, "con_rival": True, "con_contexto": True, "validacion": resumen(corridas), "epocas": epocas})
    mejor = min(s["validacion"]["deviance"]["media"] for s in seleccion)
    umbral = mejor + min(s["validacion"]["deviance"]["desv"] for s in seleccion if s["validacion"]["deviance"]["media"] == mejor)
    k = min(s["k"] for s in seleccion if s["validacion"]["deviance"]["media"] <= umbral)
    self.avisar(f"K elegido: {k}")

    # 0.1) que aporta cada bloque de la entrada
    self.avisar("Variantes con ese K")
    variantes = [s for s in seleccion if s["k"] == k]
    for con_rival, con_contexto in ((False, True), (True, False), (False, False)):
      corridas, epocas, _ = self.entrenar_varias(ds, k, con_rival, con_contexto, tr, va, va)
      variantes.append({"k": k, "con_rival": con_rival, "con_contexto": con_contexto, "validacion": resumen(corridas), "epocas": epocas})
    seleccion += variantes[1:]
    mejor_var = min(variantes, key=lambda v: v["validacion"]["deviance"]["media"])
    umbral = mejor_var["validacion"]["deviance"]["media"] + mejor_var["validacion"]["deviance"]["desv"]
    # la mas simple (menos bloques de entrada) dentro del umbral
    elegida = min((v for v in variantes if v["validacion"]["deviance"]["media"] <= umbral),
                  key=lambda v: (v["con_rival"] + v["con_contexto"], v["validacion"]["deviance"]["media"]))
    epocas_final = int(round(np.mean(elegida["epocas"])))
    self.avisar(f"Configuracion elegida: k={k}, rival={elegida['con_rival']}, contexto={elegida['con_contexto']}, {epocas_final} epocas")

    # 1-) prueba
    self.avisar("Prueba en los mundiales 2018 y 2022")
    lineas_base = {
      "medias": PredictorMedias.entrenar(ds[tr_final]),
      "mas_puntos": PredictorMasPuntos(),
      "empate": PredictorEmpate(),
    }
    prueba_resultados = {nombre: evaluar(p, ds, None, prueba) for nombre, p in lineas_base.items()}
    corridas, _, predictores = self.entrenar_varias(ds, k, elegida["con_rival"], elegida["con_contexto"], tr_final, None, prueba, epocas=epocas_final)
    prueba_resultados["lstm"] = {**resumen(corridas), "semillas": corridas}

    # 2-) guardar
    carpeta = settings.MODEL_PATH / f"lstm-{date.today().isoformat()}"
    guardar_predictor(predictores[0], carpeta, fecha=date.today().isoformat(), entrenado_hasta=2014,
                      prueba=prueba_resultados["lstm"]["semillas"][0])
    RESULTADOS.parent.mkdir(parents=True, exist_ok=True)
    RESULTADOS.write_text(json.dumps({
      "fecha": date.today().isoformat(), "dataset": DATASET.name, "partidos": len(ds),
      "particion": {"seleccion": "entrena ciclos <= 2010, valida ciclo 2014", "prueba": "entrena ciclos <= 2014, prueba mundiales 2018 y 2022 (fase mundial)"},
      "seleccion": seleccion,
      "elegida": {"k": k, "con_rival": elegida["con_rival"], "con_contexto": elegida["con_contexto"], "epocas": epocas_final},
      "prueba": prueba_resultados, "modelo": carpeta.name,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    for nombre, r in prueba_resultados.items():
      valor = (lambda m: r[m]["media"]) if nombre == "lstm" else (lambda m: r[m])
      self.avisar(f"{nombre:<11} acierto {valor('acierto_resultado'):5.1f}%  exacto {valor('marcador_exacto'):5.1f}%  "
                  f"MAE {valor('mae_goles'):.3f}  deviance {valor('deviance'):.3f}")
    self.stdout.write(self.style.SUCCESS(f"Resultados -> {RESULTADOS}; modelo -> {carpeta}"))
