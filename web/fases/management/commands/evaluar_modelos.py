import json
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from fases.dataset import EXTRAS, EXTRAS_EN_APP, construir_secuencias
from fases.evaluacion import evaluar, lote, mejor_umbral, particion
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


def deviance(seleccion):
  return seleccion["validacion"]["deviance"]["media"], seleccion["validacion"]["deviance"]["desv"]


class Command(BaseCommand):
  """
  ¿QUE HACE?
  "python manage.py evaluar_modelos" es la Fase 2 completa: elige la configuracion del LSTM con validacion temporal, lo mide una vez en los mundiales 2018 y 2022 contra las tres lineas base y guarda todo en experimentos/resultados_baseline.json (y el modelo en data/modelos/lstm-<fecha>/)

  ¿COMO LO HACE?
  0-) Seleccion: entrena con los ciclos hasta 2010 y valida con el ciclo 2014 (eliminatorias y mundial), 3 semillas por configuracion; la medida es la deviance de Poisson media

    0.0) Largo de la secuencia K en KS (con rival y con contexto); gana el K mas chico cuya deviance media este a menos de una desviacion de la mejor

    0.1) Con ese K, se prueba quitar las metricas del rival, el contexto o ambos; se queda la variante mas simple que no empeora mas de una desviacion

    0.2) Contexto adicional (EXTRAS): se prueba agregar cada uno; se agrega el que mas mejora si mejora mas que la desviacion de las semillas, y se repite con los que quedan. Solo pueden entrar los que la app sabe calcular (EXTRAS_EN_APP); los demas quedan medidos en el json

    0.3) Umbral de empate: el que maximiza el acierto de resultado en la validacion, con los modelos de la configuracion elegida

  1-) Prueba: reentrena la configuracion elegida con los ciclos hasta 2014 (epocas fijas = media de las mejores epocas de la seleccion) y la mide en los partidos de mundial de 2018 y 2022, igual que las lineas base (medias entrenadas hasta 2014, mas puntos y siempre 1-1)

  2-) Segunda prueba, el mundial 2026: la misma configuracion entrenada con los ciclos hasta 2022 y medida en sus 104 partidos, que ningun modelo vio

  3-) Guarda el modelo de la semilla 0 entrenado hasta 2022 (con su umbral de empate), que es el que usa la app, y el json de resultados

  Tarda unos 20 minutos en una CPU de 4 nucleos; imprime el avance de cada entrenamiento
  """
  help = "Fase 2: seleccion del LSTM, prueba contra lineas base y resultados en experimentos/resultados_baseline.json"
  requires_system_checks = []

  def avisar(self, texto):
    self.stdout.write(texto)
    self.stdout.flush()

  def entrenar_varias(self, ds, k, con_rival, con_contexto, filas_tr, filas_va, filas_ev, epocas=None, extras=()):
    """Entrena una configuracion con cada semilla; devuelve (metricas por semilla, mejores epocas, predictores, secuencias)"""
    sec = construir_secuencias(ds, k=k, con_rival=con_rival, con_contexto=con_contexto, extras=extras)
    corridas, epocas_mejores, predictores = [], [], []
    for semilla in SEMILLAS:
      inicio = time.time()
      validacion = filas_va if epocas is None else None
      predictor, historia = PredictorKeras.entrenar(ds, sec, filas_tr, validacion, k, con_rival, con_contexto, semilla=semilla,
                                                    epocas=epocas or 80, extras=extras)
      if epocas is None:
        epocas_mejores.append(int(np.argmin(historia["val_loss"])) + 1)
      resultado = evaluar(predictor, ds, sec, filas_ev)
      corridas.append(resultado)
      predictores.append(predictor)
      self.avisar(f"  k={k:<2} rival={con_rival!s:<5} contexto={con_contexto!s:<5} extras={','.join(extras) or '-':<26} semilla={semilla}: "
                  f"deviance {resultado['deviance']:.4f}, acierto {resultado['acierto_resultado']:.1f}% ({time.time() - inicio:.0f}s)")
    return corridas, epocas_mejores, predictores, sec

  def configuracion(self, ds, tr, va, k, con_rival, con_contexto, extras=()):
    corridas, epocas, predictores, sec = self.entrenar_varias(ds, k, con_rival, con_contexto, tr, va, va, extras=extras)
    return {"k": k, "con_rival": con_rival, "con_contexto": con_contexto, "extras": list(extras),
            "validacion": resumen(corridas), "epocas": epocas}, predictores, sec

  def handle(self, *args, **opciones):
    if not DATASET.exists():
      raise CommandError(f"No existe {DATASET}: ejecutar antes 'python manage.py construir_dataset'")
    ds = pd.read_csv(DATASET)
    faltan = [c.format(i=0) for e in EXTRAS for c in EXTRAS[e] if c.format(i=0) not in ds]
    if faltan:
      raise CommandError(f"El dataset no tiene el contexto {faltan}: regenerarlo con 'python manage.py construir_dataset'")
    tr, va, _ = particion(ds, hasta=2010, validacion=2014)
    tr_final, _, prueba = particion(ds, hasta=2014, prueba=[2018, 2022])
    self.avisar(f"Dataset: {len(ds)} partidos; seleccion {tr.sum()} entrenamiento / {va.sum()} validacion; prueba {prueba.sum()} partidos de mundial")

    # 0.0) largo de la secuencia
    self.avisar("Seleccion de K (con rival y contexto)")
    seleccion = [self.configuracion(ds, tr, va, k, True, True)[0] for k in KS]
    mejor, desv = min(deviance(s) for s in seleccion)
    k = min(s["k"] for s in seleccion if deviance(s)[0] <= mejor + desv)
    self.avisar(f"K elegido: {k}")

    # 0.1) que aporta cada bloque de la entrada
    self.avisar("Variantes con ese K")
    variantes = [s for s in seleccion if s["k"] == k]
    for con_rival, con_contexto in ((False, True), (True, False), (False, False)):
      variantes.append(self.configuracion(ds, tr, va, k, con_rival, con_contexto)[0])
    seleccion += variantes[1:]
    mejor_var = min(variantes, key=lambda v: deviance(v)[0])
    umbral = sum(deviance(mejor_var))
    elegida = min((v for v in variantes if deviance(v)[0] <= umbral), key=lambda v: (v["con_rival"] + v["con_contexto"], deviance(v)[0]))
    self.avisar(f"Variante elegida: rival={elegida['con_rival']}, contexto={elegida['con_contexto']}")

    # 0.2) contexto adicional, de a uno
    self.avisar("Contexto adicional")
    base = dict(elegida)
    _, predictores_base, sec_base = self.configuracion(ds, tr, va, k, base["con_rival"], base["con_contexto"])
    pruebas_extras = []
    while True:
      candidatos = [e for e in EXTRAS if e not in base["extras"] and (not pruebas_extras or e in EXTRAS_EN_APP)]
      ronda = []
      for e in candidatos:
        resultado, predictores, sec = self.configuracion(ds, tr, va, k, base["con_rival"], base["con_contexto"], tuple(base["extras"] + [e]))
        ronda.append((e, resultado, predictores, sec))
        pruebas_extras.append(resultado)
      seleccion += [r for _, r, _, _ in ronda]
      en_app = [x for x in ronda if x[0] in EXTRAS_EN_APP]
      if not en_app:
        break
      e, resultado, predictores, sec = min(en_app, key=lambda x: deviance(x[1])[0])
      mejora = deviance(base)[0] - deviance(resultado)[0]
      if mejora <= max(deviance(base)[1], deviance(resultado)[1]):
        break
      self.avisar(f"  se agrega {e}: deviance {deviance(base)[0]:.4f} -> {deviance(resultado)[0]:.4f}")
      base, predictores_base, sec_base = resultado, predictores, sec
    epocas_final = int(round(np.mean(base["epocas"])))
    self.avisar(f"Configuracion elegida: k={k}, rival={base['con_rival']}, contexto={base['con_contexto']}, extras={base['extras']}, {epocas_final} epocas")

    # 0.3) umbral de empate con los modelos de la seleccion (entrenados hasta 2010) sobre la validacion
    goles_va = ds.loc[va, ["goles_0", "goles_1"]].to_numpy()
    umbral_empate, curva = mejor_umbral(goles_va, [p.predecir(lote(ds, sec_base, va)) for p in predictores_base])
    self.avisar(f"Umbral de empate: {umbral_empate:.2f} (acierto en validacion {curva[0] * 100:.1f}% sin umbral -> {max(curva) * 100:.1f}%)")

    # 1-) prueba
    self.avisar("Prueba en los mundiales 2018 y 2022")
    lineas_base = {
      "medias": PredictorMedias.entrenar(ds[tr_final]),
      "mas_puntos": PredictorMasPuntos(),
      "empate": PredictorEmpate(),
    }
    prueba_resultados = {nombre: evaluar(p, ds, None, prueba) for nombre, p in lineas_base.items()}
    sec = construir_secuencias(ds, k=k, con_rival=base["con_rival"], con_contexto=base["con_contexto"], extras=base["extras"])
    corridas, sin_umbral, predictores = [], [], []
    for semilla in SEMILLAS:
      predictor, _ = PredictorKeras.entrenar(ds, sec, tr_final, None, k, base["con_rival"], base["con_contexto"], semilla=semilla,
                                             epocas=epocas_final, extras=base["extras"])
      sin_umbral.append(evaluar(predictor, ds, sec, prueba))
      predictor.config["umbral_empate"] = umbral_empate
      corridas.append(evaluar(predictor, ds, sec, prueba))
      predictores.append(predictor)
      self.avisar(f"  semilla={semilla}: acierto {sin_umbral[-1]['acierto_resultado']:.1f}% sin umbral, {corridas[-1]['acierto_resultado']:.1f}% con umbral")
    prueba_resultados["lstm"] = {**resumen(corridas), "semillas": corridas}
    prueba_resultados["lstm_sin_umbral"] = {**resumen(sin_umbral), "semillas": sin_umbral}

    # 2-) mundial 2026, entrenando con todo hasta 2022
    tr_2026, _, prueba_2026 = particion(ds, hasta=2022, prueba=[2026])
    resultados_2026 = {}
    if prueba_2026.any():
      self.avisar(f"Prueba en el mundial 2026 ({prueba_2026.sum()} partidos), entrenando hasta 2022")
      resultados_2026 = {nombre: evaluar(p, ds, None, prueba_2026) for nombre, p in
                         {"medias": PredictorMedias.entrenar(ds[tr_2026]), "mas_puntos": PredictorMasPuntos(), "empate": PredictorEmpate()}.items()}
      corridas_2026, predictores = [], []
      for semilla in SEMILLAS:
        predictor, _ = PredictorKeras.entrenar(ds, sec, tr_2026, None, k, base["con_rival"], base["con_contexto"], semilla=semilla,
                                               epocas=epocas_final, extras=base["extras"])
        predictor.config["umbral_empate"] = umbral_empate
        corridas_2026.append(evaluar(predictor, ds, sec, prueba_2026))
        predictores.append(predictor)
        self.avisar(f"  semilla={semilla}: acierto {corridas_2026[-1]['acierto_resultado']:.1f}%")
      resultados_2026["lstm"] = {**resumen(corridas_2026), "semillas": corridas_2026}

    # 2-) guardar
    carpeta = settings.MODEL_PATH / f"lstm-{date.today().isoformat()}"
    guardar_predictor(predictores[0], carpeta, fecha=date.today().isoformat(), entrenado_hasta=2022 if resultados_2026 else 2014,
                      prueba=(resultados_2026 or prueba_resultados)["lstm"]["semillas"][0])
    RESULTADOS.parent.mkdir(parents=True, exist_ok=True)
    RESULTADOS.write_text(json.dumps({
      "fecha": date.today().isoformat(), "dataset": DATASET.name, "partidos": len(ds),
      "particion": {"seleccion": "entrena ciclos <= 2010, valida ciclo 2014", "prueba": "entrena ciclos <= 2014, prueba mundiales 2018 y 2022 (fase mundial)"},
      "seleccion": seleccion,
      "elegida": {"k": k, "con_rival": base["con_rival"], "con_contexto": base["con_contexto"], "extras": base["extras"],
                  "epocas": epocas_final, "umbral_empate": umbral_empate},
      "umbral_empate": {"umbrales": [round(u, 2) for u in np.arange(0, 0.41, 0.01)], "acierto_validacion": curva},
      "prueba": prueba_resultados, "prueba_2026": resultados_2026, "modelo": carpeta.name,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    for titulo, tabla in (("2018 y 2022", prueba_resultados), ("2026", resultados_2026)):
      self.avisar(f"Prueba {titulo}:")
      for nombre, r in tabla.items():
        valor = (lambda m: r[m]["media"]) if nombre.startswith("lstm") else (lambda m: r[m])
        self.avisar(f"  {nombre:<16} acierto {valor('acierto_resultado'):5.1f}%  exacto {valor('marcador_exacto'):5.1f}%  "
                    f"MAE {valor('mae_goles'):.3f}  deviance {valor('deviance'):.3f}")
    self.stdout.write(self.style.SUCCESS(f"Resultados -> {RESULTADOS}; modelo -> {carpeta}"))
