import json
import tempfile
from pathlib import Path

import pandas as pd
from django.test import SimpleTestCase

from .dataset import CATEGORIAS, CICLO_INTERNACIONALES, construir_dataset, construir_secuencias, mapa_paises
from .modulo import ALL_METRICS, COLS_MODELO_POR_EQUIPO, METRICAS_ABSTRACTAS, historico_abstractas
from .predictores import Lote, PredictorMedias, cargar_predictor, guardar_predictor, lote_desde_entrada


def partidos_df(filas):
  return pd.DataFrame(filas, columns=["home", "away", "score_0", "score_1", "agno"])


PROXY_VACIO = pd.DataFrame(columns=["pais", "agno", *METRICAS_ABSTRACTAS])


class ConstruirDatasetTests(SimpleTestCase):
  """Torneo inventado: eliminatorias A-B 2:0 y B-C 1:1, mundial A-C 1:0 (ids: A=0, B=1, C=2)"""

  def setUp(self):
    self.clasificaciones = partidos_df([["A", "B", 2, 0, 1934], ["B", "C", 1, 1, 1934]])
    self.partidos = partidos_df([["A", "C", 1, 0, 1934]])
    self.ds = construir_dataset(self.partidos, self.clasificaciones, PROXY_VACIO, agnos=[1934])

  def test_una_fila_por_partido_en_orden(self):
    self.assertEqual(self.ds["fase"].tolist(), ["eliminatoria", "eliminatoria", "mundial"])
    self.assertEqual(self.ds[["goles_0", "goles_1"]].values.tolist(), [[2, 0], [1, 1], [1, 0]])

  def test_metricas_previas_al_partido(self):
    # 1er partido: nadie tiene historia, ni siquiera su propio resultado
    self.assertEqual(self.ds.loc[0, ["PJ_0", "GF_0", "PJ_1", "GF_1"]].tolist(), [0, 0, 0, 0])
    # B llega al 2do partido con la derrota 0:2
    self.assertEqual(self.ds.loc[1, ["PTS_0", "PJ_0", "PP_0", "GF_0", "GC_0", "D_0"]].tolist(), [0, 1, 1, 0, 2, -2])
    # al mundial: A con la victoria 2:0 y C con el empate 1:1
    self.assertEqual(self.ds.loc[2, ["PTS_0", "PJ_0", "PG_0", "GF_0", "GC_0", "D_0"]].tolist(), [3, 1, 1, 2, 0, 2])
    self.assertEqual(self.ds.loc[2, ["PTS_1", "PJ_1", "PE_1", "GF_1", "GC_1", "D_1"]].tolist(), [1, 1, 1, 1, 1, 0])

  def test_metricas_derivadas(self):
    # rate_GF = GF / PJ * 10
    self.assertAlmostEqual(self.ds.loc[2, "rate_GF_0"], 20.0)
    self.assertAlmostEqual(self.ds.loc[2, "rate_GF_1"], 10.0)
    # el progreso abstracto vuelve al historico al empezar el mundial (como en la app): sin datos VI = PKATTALLOW = 0.1
    # gkps = (VI/PJ)*60 + 40 - ((GC - PKATTALLOW)/PJ)*10 = 6 + 40 + 1
    self.assertAlmostEqual(self.ds.loc[2, "gkps_0"], 47.0)

  def test_equipo_sin_eliminatorias_hereda_su_ultimo_ciclo(self):
    clasif = partidos_df([["A", "B", 2, 0, 1934], ["B", "C", 1, 1, 1938]])
    partidos = partidos_df([["A", "C", 1, 0, 1934], ["A", "B", 0, 0, 1938]])
    ds = construir_dataset(partidos, clasif, PROXY_VACIO, agnos=[1934, 1938])
    # en 1938 A no jugo eliminatorias: arranca con lo de sus eliminatorias de 1934 (victoria 2:0), no con el mundial
    fila = ds[(ds["agno"] == 1938) & (ds["fase"] == "mundial")].iloc[0]
    self.assertEqual([fila["PJ_0"], fila["GF_0"]], [1, 2])


class HistoricoAbstractasTests(SimpleTestCase):

  def test_solo_mundiales_anteriores(self):
    proxy = pd.DataFrame(
      [["A", 1930, 2, 1, 0, 0], ["A", 1938, 5, 1, 0, 0], ["B", 1934, 1, 1, 0, 0]],
      columns=["pais", "agno", *METRICAS_ABSTRACTAS],
    )
    hist = historico_abstractas(proxy, {"A": 0, "B": 1}, agno=1934)
    # A solo suma 1930; B (1934) no entra porque 1934 no es anterior a 1934
    self.assertEqual(hist["pais"].tolist(), [0])
    self.assertEqual(hist.loc[0, "VI"], 2)


class ConstruirSecuenciasTests(SimpleTestCase):

  def test_relleno_y_orden(self):
    clasif = partidos_df([["A", "B", 2, 0, 1934], ["B", "C", 1, 1, 1934]])
    partidos = partidos_df([["A", "C", 1, 0, 1934]])
    ds = construir_dataset(partidos, clasif, PROXY_VACIO, agnos=[1934])
    sec = construir_secuencias(ds, k=3)
    pj = 1  # posicion de PJ dentro de ALL_METRICS
    # A en el partido del mundial: 1 paso de relleno, luego su eliminatoria (PJ 0) y el actual (PJ 1)
    self.assertEqual(sec["mascara_0"][2].tolist(), [0, 1, 1])
    self.assertEqual(sec["seq_0"][2, :, pj].tolist(), [0, 0, 1])
    self.assertEqual(sec["y"][2].tolist(), [1, 0])
    self.assertEqual(sec["id_0"][2], mapa_paises(partidos, clasif)["A"])

  def test_metricas_del_rival(self):
    clasif = partidos_df([["A", "B", 2, 0, 1934], ["B", "C", 1, 1, 1934]])
    partidos = partidos_df([["A", "C", 1, 0, 1934]])
    ds = construir_dataset(partidos, clasif, PROXY_VACIO, agnos=[1934])
    f = len(ALL_METRICS)
    sec = construir_secuencias(ds, k=2, con_contexto=False)
    self.assertEqual(sec["seq_0"].shape, (3, 2, 2 * f))
    # en el partido del mundial, el ultimo paso de A trae sus metricas y despues las de C (su rival ese dia)
    self.assertEqual(sec["seq_0"][2, -1, :f].tolist(), ds.loc[2, [f"{m}_0" for m in ALL_METRICS]].tolist())
    self.assertEqual(sec["seq_0"][2, -1, f:].tolist(), ds.loc[2, [f"{m}_1" for m in ALL_METRICS]].tolist())
    # y el de C, al reves
    self.assertEqual(sec["seq_1"][2, -1, f:].tolist(), ds.loc[2, [f"{m}_0" for m in ALL_METRICS]].tolist())
    self.assertEqual(construir_secuencias(ds, k=2, con_rival=False, con_contexto=False)["seq_0"].shape, (3, 2, f))

  def test_contexto(self):
    clasif = partidos_df([["A", "B", 2, 0, 1934]])
    partidos = partidos_df([["A", "B", 1, 0, 1934]])
    ds = construir_dataset(partidos, clasif, PROXY_VACIO, agnos=[1934])
    sec = construir_secuencias(ds, k=1)
    contexto = sec["seq_0"][:, -1, 2 * len(ALL_METRICS):]
    mundial = [0.0] * len(CATEGORIAS)
    mundial[CATEGORIAS.index("mundial")] = 1.0
    eliminatoria = [0.0] * len(CATEGORIAS)
    eliminatoria[CATEGORIAS.index("eliminatoria_mundial")] = 1.0
    # eliminatoria en casa: es_local 1, neutral 0; mundial sin dato de sede: neutral 1, nadie es local
    self.assertEqual(contexto[0].tolist(), [1.0, 0.0, *eliminatoria])
    self.assertEqual(contexto[1].tolist(), [0.0, 1.0, *mundial])
    # el visitante nunca es local
    self.assertEqual(sec["seq_1"][0, -1, 2 * len(ALL_METRICS)], 0.0)


class PredictoresTests(SimpleTestCase):

  def setUp(self):
    clasif = partidos_df([["A", "B", 2, 0, 1934], ["B", "C", 1, 1, 1934]])
    partidos = partidos_df([["A", "C", 1, 0, 1934]])
    self.ds = construir_dataset(partidos, clasif, PROXY_VACIO, agnos=[1934])

  def test_medias(self):
    predictor = PredictorMedias.entrenar(self.ds)
    salida = predictor.predecir(Lote(self.ds))
    # medias de (2, 1, 1) y (0, 1, 0)
    self.assertEqual(salida.shape, (3, 2))
    self.assertAlmostEqual(salida[0, 0], 4 / 3)
    self.assertAlmostEqual(salida[0, 1], 1 / 3)

  def test_guardar_y_cargar(self):
    with tempfile.TemporaryDirectory() as carpeta:
      guardar_predictor(PredictorMedias(1.5, 1.0), carpeta, fecha="2026-10-15")
      predictor, esquema = cargar_predictor(carpeta)
      self.assertEqual(predictor.predecir(Lote(self.ds)).tolist()[0], [1.5, 1.0])
      self.assertEqual(esquema["fecha"], "2026-10-15")

  def test_rechaza_metricas_distintas(self):
    with tempfile.TemporaryDirectory() as carpeta:
      guardar_predictor(PredictorMedias(1.5, 1.0), carpeta)
      ruta = Path(carpeta) / "esquema.json"
      esquema = json.loads(ruta.read_text(encoding="utf-8"))
      esquema["metricas"] = esquema["metricas"][:-1]
      ruta.write_text(json.dumps(esquema), encoding="utf-8")
      with self.assertRaisesRegex(ValueError, "reentrenarlo"):
        cargar_predictor(carpeta)

  def test_lote_desde_entrada(self):
    # dos partidos con la forma de "calculo_metricas_0": [home, ALL_METRICS_0, score_0, away, ALL_METRICS_1, score_1]
    lado = lambda equipo, base: [equipo, *[base + j for j in range(len(ALL_METRICS))], 9]
    X = [lado(3, 0) + lado(7, 100), lado(4, 0) + lado(8, 100)]
    self.assertEqual(len(X[0]), 2 * COLS_MODELO_POR_EQUIPO)
    tabla = lote_desde_entrada(X).tabla
    self.assertEqual(tabla["home_id"].tolist(), [3, 4])
    self.assertEqual(tabla["away_id"].tolist(), [7, 8])
    self.assertEqual(tabla[f"{ALL_METRICS[1]}_1"].tolist(), [101, 101])
    self.assertNotIn("score_0", tabla.columns)
    # la app la puede pasar tal cual a un predictor
    self.assertEqual(PredictorMedias(1.5, 1.0).predecir(lote_desde_entrada(X)).shape, (2, 2))


class CicloInternacionalesTests(SimpleTestCase):

  def test_ciclo_por_fechas(self):
    internacionales = pd.DataFrame([
      ["1930-07-13", 1930, "A", "B", 1, 0, "mundial", 1],    # mundial 1930: marca el inicio del ciclo de 1934
      ["1931-05-01", 1931, "A", "C", 3, 0, "amistoso", 0],
      ["1933-06-01", 1933, "B", "C", 2, 2, "eliminatoria_mundial", 0],
      ["1934-05-27", 1934, "A", "B", 2, 1, "mundial", 0],    # anfitrion: no neutral
      ["1934-08-01", 1934, "A", "C", 9, 0, "amistoso", 0],   # despues del mundial: no entra en su ciclo
    ], columns=["fecha", "agno", "home", "away", "score_0", "score_1", "categoria", "neutral"])
    partidos = partidos_df([["A", "B", 2, 1, 1934]])
    ds = construir_dataset(partidos, partidos_df([]), PROXY_VACIO, agnos=[1930, 1934],
                           internacionales=internacionales, ciclo=CICLO_INTERNACIONALES)
    ciclo_1934 = ds[(ds["agno"] == 1934) & (ds["fase"] == "eliminatoria")]
    self.assertEqual(ciclo_1934["categoria"].tolist(), ["amistoso", "eliminatoria_mundial"])
    mundial_1934 = ds[(ds["agno"] == 1934) & (ds["fase"] == "mundial")].iloc[0]
    self.assertEqual(mundial_1934["neutral"], 0)
    # A llega al mundial con su amistoso ganado 3-0
    self.assertEqual(mundial_1934["GF_0"], 3)
