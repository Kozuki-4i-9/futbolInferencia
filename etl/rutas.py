"""Carpetas del proyecto para los scripts de etl/: se calculan desde este archivo, asi no dependen de la PC donde se ejecutan"""
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
DATA = RAIZ / "data"
GRUPOS = DATA / "grupos"
PARTIDOS = DATA / "partidos"
PROXY_DESEMPEGNO = DATA / "proxy_desempegno"
