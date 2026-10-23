from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand

from fases.dataset import CICLO_ELIMINATORIAS, CICLO_INTERNACIONALES, cargar_fuentes, construir_dataset

DATASETS = Path(settings.BASE_DIR).parent / "data" / "datasets"
SALIDAS = {
  CICLO_ELIMINATORIAS: DATASETS / "partidos_features.csv",
  CICLO_INTERNACIONALES: DATASETS / "partidos_features_internacionales.csv",
}


class Command(BaseCommand):
  """
  ¿QUE HACE?
  "python manage.py construir_dataset" genera la tabla de entrenamiento (una fila por partido real con las metricas previas de ambos equipos, sus goles, el tipo de partido y si fue en campo neutral) y la guarda en data/datasets/. Tarda unos minutos; el entrenamiento lee el csv en vez de recalcularla

  ¿COMO LO HACE?
  0-) Lee partidos, eliminatorias, metricas abstractas y partidos internacionales de la BD con "cargar_fuentes"

  1-) Arma la tabla con "construir_dataset" segun --ciclo y la escribe en su archivo de (SALIDAS), creando la carpeta si no existe. El archivo no se versiona: se reconstruye con este comando

  Args (linea de comandos):

    - --ciclo eliminatorias (por defecto, lo que usa la app hoy) o internacionales (todos los partidos de selecciones entre mundiales)
  """
  help = "Genera el dataset de entrenamiento en data/datasets/"
  requires_system_checks = []

  def add_arguments(self, parser):
    parser.add_argument("--ciclo", choices=sorted(SALIDAS), default=CICLO_ELIMINATORIAS,
                        help="partidos previos a cada mundial: solo eliminatorias o todos los internacionales")

  def handle(self, *args, **opciones):
    partidos, clasificaciones, proxy, internacionales, fixtures = cargar_fuentes()
    dataset = construir_dataset(partidos, clasificaciones, proxy, internacionales=internacionales, ciclo=opciones["ciclo"], fixtures=fixtures)
    salida = SALIDAS[opciones["ciclo"]]
    salida.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(salida, index=False, encoding="utf-8")
    self.stdout.write(self.style.SUCCESS(f"{len(dataset)} partidos -> {salida}"))
