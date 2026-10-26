from datetime import date
from pathlib import Path

import pandas as pd
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from fases.predictores import PredictorMedias, guardar_predictor

DATASET = Path(settings.BASE_DIR).parent / "data" / "datasets" / "partidos_features.csv"


class Command(BaseCommand):
  """
  ¿QUE HACE?
  "python manage.py entrenar_predictor" entrena la linea base "PredictorMedias" con la tabla de "construir_dataset" y la guarda en data/modelos/medias/ (o en --carpeta), lista para que la app la cargue con MODEL_DIR

  ¿COMO LO HACE?
  0-) Lee (DATASET); si no existe pide ejecutar antes "construir_dataset"

  1-) Si se pasa --hasta, deja solo los partidos con agno <= ese año (para evaluar despues con mundiales que el predictor no vio)

  2-) Entrena con "PredictorMedias.entrenar" y guarda con "guardar_predictor" su esquema.json con fecha, años usados y numero de partidos
  """
  help = "Entrena la linea base de medias y la guarda en data/modelos/"
  requires_system_checks = []

  def add_arguments(self, parser):
    parser.add_argument("--hasta", type=int, default=None, help="ultimo año de entrenamiento (por defecto todos)")
    parser.add_argument("--carpeta", default="medias", help="carpeta de salida dentro de data/modelos/")

  def handle(self, *args, **opciones):
    if not DATASET.exists():
      raise CommandError(f"No existe {DATASET}: ejecutar antes 'python manage.py construir_dataset'")
    dataset = pd.read_csv(DATASET)
    if opciones["hasta"] is not None:
      dataset = dataset[dataset["agno"] <= opciones["hasta"]]

    predictor = PredictorMedias.entrenar(dataset)
    carpeta = settings.MODEL_PATH / opciones["carpeta"]
    guardar_predictor(predictor, carpeta, fecha=date.today().isoformat(),
                      agnos=[int(dataset["agno"].min()), int(dataset["agno"].max())], partidos=len(dataset))
    self.stdout.write(self.style.SUCCESS(
      f"{predictor.nombre}: home {predictor.media_0:.3f}, away {predictor.media_1:.3f} ({len(dataset)} partidos) -> {carpeta}"))
