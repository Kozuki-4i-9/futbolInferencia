from pathlib import Path

import pandas as pd
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from fases.models import partidos, fixtures, grupos, clasificaciones, proxy_desempegno

DATA_DIR = Path(settings.BASE_DIR).parent / "data"

# modelo -> (csv relativo a DATA_DIR, renombres columna_csv -> campo_modelo)
TABLAS = {
  partidos: ("partidos/partidos.csv", {"year": "agno"}),
  fixtures: ("partidos/fixtures.csv", {}),
  grupos: ("grupos/grupos_mundiales.csv", {}),
  clasificaciones: ("partidos/clasificaciones.csv", {}),
}

# salida de etl/busca_metricas.py (fbref): un csv por mundial con pais, VI, SOT, PKATT, PKATTALLOW
PROXY_DIR = DATA_DIR / "proxy_desempegno"
# copia de METRICAS_ABSTRACTAS: importar modulo.py arrastra tensorflow y aqui no hace falta
PROXY_ABSTRACTAS = ["VI", "SOT", "PKATT", "PKATTALLOW"]


def campos_de(modelo):
  """Campos de la tabla que se rellenan desde el csv (todos menos la clave primaria)"""
  return [f.name for f in modelo._meta.concrete_fields if not f.primary_key]


def leer_csv(modelo, ruta, renombres):
  """
  ¿QUE HACE?
  Lee el csv de una tabla y lo deja con exactamente las columnas del modelo de Django, listo para "bulk_create"

  ¿COMO LO HACE?
  0-) Lee todo como texto (los campos del modelo son CharField) sin convertir celdas vacias en NaN

  1-) Aplica (renombres) para alinear nombres del csv con los del modelo ("year" -> "agno")

  2-) Si falta algun campo del modelo lanza (CommandError) con la lista completa; las columnas sobrantes ("id", indices de pandas) se descartan

  Args:

    - modelo: clase del modelo de Django destino

    - ruta: Path del csv

    - renombres: diccionario columna_csv -> campo_modelo

  Returns:

    - df: dataframe con las columnas de (campos_de(modelo)) en ese orden
  """
  if not ruta.exists():
    raise CommandError(f"No existe {ruta}")
  df = pd.read_csv(ruta, dtype=str, keep_default_na=False, encoding="utf-8").rename(columns=renombres)
  campos = campos_de(modelo)
  faltan = [c for c in campos if c not in df.columns]
  if faltan:
    raise CommandError(f"{ruta.name}: faltan columnas {faltan} para {modelo.__name__}")
  return df[campos]


def leer_proxy():
  """
  ¿QUE HACE?
  Arma las filas de la tabla (proxy_desempegno), una por pais y mundial, a partir de los csv por mundial del scraper. Se guarda el año para que cada mundial solo use los datos de mundiales anteriores (sin fuga de datos del futuro)

  ¿COMO LO HACE?
  0-) Busca "metricas_abstractas_<año>.csv" en (PROXY_DIR); si no hay ninguno devuelve None y la tabla queda vacia

  1-) A cada csv le agrega la columna "agno" tomada del nombre del archivo y los concatena

  2-) Una celda sin dato real queda "" y no 0, para que "formar_dataset_real" la lea como NaN

  3-) gkps, mds, mos y mms quedan "" porque se derivan en "calculo_metricas_0", no se guardan

  Returns:

    - df con las columnas de (campos_de(proxy_desempegno)), o None si no hay csv
  """
  rutas = sorted(PROXY_DIR.glob("metricas_abstractas_*.csv"))
  if not rutas:
    return None
  df = pd.concat(
    [pd.read_csv(r, encoding="utf-8").assign(agno=r.stem.rsplit("_", 1)[-1]) for r in rutas],
    ignore_index=True,
  )
  df[PROXY_ABSTRACTAS] = df[PROXY_ABSTRACTAS].apply(pd.to_numeric, errors="coerce")
  df = df.astype(object).where(df.notna(), "").astype(str)
  for campo in campos_de(proxy_desempegno):
    if campo not in df.columns:
      df[campo] = ""
  return df[campos_de(proxy_desempegno)]


class Command(BaseCommand):
  """
  ¿QUE HACE?
  "python manage.py load_data" llena las tablas de la app con los csv de data/, para reconstruir la BD desde cero en cualquier maquina (o al arrancar el contenedor de Docker en la Fase 5)

  ¿COMO LO HACE?
  0-) No ejecuta los system checks (requires_system_checks = []): cargar urls -> views -> modulo arrastra tensorflow, y no hace falta para cargar datos

  1-) Lee y valida TODOS los csv antes de escribir nada; si uno falla no se toca la BD

  2-) Dentro de una sola transaccion, por cada tabla: si ya tiene filas y no se paso --reemplazar la salta (asi se puede ejecutar en cada arranque sin duplicar); con --reemplazar la vacia y la recarga con "bulk_create"

  Args (linea de comandos):

    - --reemplazar: borra el contenido actual de las tablas antes de cargar
  """
  help = "Carga los CSV de data/ en las tablas de la app fases"
  requires_system_checks = []

  def add_arguments(self, parser):
    parser.add_argument("--reemplazar", action="store_true", help="Vacia las tablas antes de cargarlas")

  def handle(self, *args, **opciones):
    if not DATA_DIR.exists():
      raise CommandError(f"No existe la carpeta de datos {DATA_DIR}")

    cargas = [(modelo, leer_csv(modelo, DATA_DIR / csv, renombres)) for modelo, (csv, renombres) in TABLAS.items()]
    proxy = leer_proxy()
    if proxy is None:
      self.stdout.write(self.style.WARNING(f"Sin csv en {PROXY_DIR}: proxy_desempegno queda sin cargar"))
    else:
      cargas.append((proxy_desempegno, proxy))

    with transaction.atomic():
      for modelo, df in cargas:
        self.cargar(modelo, df, opciones["reemplazar"])

  def cargar(self, modelo, df, reemplazar):
    """Carga (df) en (modelo) respetando la regla de --reemplazar descrita en la clase"""
    nombre = modelo._meta.db_table
    existentes = modelo.objects.count()
    if existentes and not reemplazar:
      self.stdout.write(self.style.WARNING(f"{nombre}: ya tiene {existentes} filas, se omite (usa --reemplazar)"))
      return
    if reemplazar:
      modelo.objects.all().delete()
    modelo.objects.bulk_create([modelo(**fila) for fila in df.to_dict("records")], batch_size=1000)
    self.stdout.write(self.style.SUCCESS(f"{nombre}: {len(df)} filas cargadas"))
