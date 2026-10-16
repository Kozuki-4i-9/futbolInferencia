from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand

from fases.dataset import cargar_fuentes, construir_dataset

SALIDA = Path(settings.BASE_DIR).parent / "data" / "datasets" / "partidos_features.csv"


class Command(BaseCommand):
  """
  ¿QUE HACE?
  "python manage.py construir_dataset" genera la tabla de entrenamiento (una fila por partido real con las metricas previas de ambos equipos y sus goles) y la guarda en data/datasets/. Tarda unos minutos; el entrenamiento lee el csv en vez de recalcularla

  ¿COMO LO HACE?
  0-) Lee partidos, eliminatorias y metricas abstractas de la BD con "cargar_fuentes"

  1-) Arma la tabla con "construir_dataset" y la escribe en (SALIDA), creando la carpeta si no existe. El archivo no se versiona: se reconstruye con este comando
  """
  help = "Genera data/datasets/partidos_features.csv para entrenar"
  requires_system_checks = []

  def handle(self, *args, **opciones):
    dataset = construir_dataset(*cargar_fuentes())
    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(SALIDA, index=False, encoding="utf-8")
    self.stdout.write(self.style.SUCCESS(f"{len(dataset)} partidos -> {SALIDA}"))
