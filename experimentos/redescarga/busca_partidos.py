"""
Descarga de Wikipedia (es) todos los partidos de un mundial (cada partido es una tabla "vevent") y los guarda en
data/redescarga/partidos/partidos_<año>.csv, con el marcador del partido separado de la tanda de penales.

A diferencia de etl/busca_partidos_0.py, que reemplazaba el marcador por el de los penales
("3:3 (4:2 p.)" quedaba como "4:2"), aqui cada fila guarda:

    fecha, home, score_0, score_1, away, prorroga, penales_0, penales_1, year

Uso:
    python experimentos/redescarga/busca_partidos.py --anio 2022
    python experimentos/redescarga/busca_partidos.py --todos
"""
import argparse
import re
import sys
from pathlib import Path

import pandas as pd
from selenium.webdriver.common.by import By

# reutiliza el navegador y la URL del scraper de grupos (etl/busca_grupos.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "etl"))
from busca_grupos import URL, crear_driver  # noqa: E402

SALIDA = Path(__file__).resolve().parents[2] / "data" / "redescarga" / "partidos"
AGNOS = [1930, 1934, 1938, 1950, 1954, 1958, 1962, 1966, 1970, 1974, 1978, 1982, 1986, 1990, 1994, 1998, 2002, 2006, 2010, 2014, 2018, 2022]

MARCADOR = re.compile(r"(\d+)\s*[:\-–]\s*(\d+)")
PENALES = re.compile(r"\((\d+)\s*[:\-–]\s*(\d+)\s*p\.?\)")
PRORROGA = re.compile(r"t\.\s*s\.|pr[oó]rroga|a\.\s*e\.\s*t\.", re.IGNORECASE)
# enlace del equipo: el primero que no sea la bandera
EQUIPO = ".//a[not(ancestor::span[contains(@class, 'flagicon')])][1]"


def leer_partido(tabla):
    """
    ¿QUE HACE?
    Lee una tabla "vevent" de Wikipedia: fecha, equipos, marcador, prorroga y penales

    ¿COMO LO HACE?
    0-) La 1ra celda de la 1ra fila es la fecha; home esta en la celda de ancho 24% y away en la de 22%

    1-) La celda entre ambas trae el marcador: el primer "a:b" es el resultado del partido (90' o 120'); "t. s." marca prorroga y "(a:b p.)" la tanda de penales

    Returns:

      - diccionario con fecha, home, score_0, score_1, away, prorroga, penales_0, penales_1; None si la tabla no tiene dos equipos y un marcador (partidos anulados o no jugados)
    """
    fila = tabla.find_elements(By.XPATH, ".//tr[1]/td")
    local = tabla.find_elements(By.XPATH, ".//tr[1]/td[@width='24%']")
    visita = tabla.find_elements(By.XPATH, ".//tr[1]/td[@width='22%']")
    if not (fila and local and visita):
        return None
    celdas = [c for c in fila]
    i_local, i_visita = celdas.index(local[0]), celdas.index(visita[0])
    texto_marcador = " ".join(c.text for c in celdas[i_local + 1:i_visita])

    marcador = MARCADOR.search(texto_marcador)
    enlaces_local = local[0].find_elements(By.XPATH, EQUIPO)
    enlaces_visita = visita[0].find_elements(By.XPATH, EQUIPO)
    if not (marcador and enlaces_local and enlaces_visita):
        return None
    penales = PENALES.search(texto_marcador)
    return {
        "fecha": celdas[0].text.strip(),
        "home": enlaces_local[0].text.strip(),
        "score_0": int(marcador.group(1)),
        "score_1": int(marcador.group(2)),
        "away": enlaces_visita[0].text.strip(),
        "prorroga": bool(PRORROGA.search(texto_marcador)),
        "penales_0": int(penales.group(1)) if penales else None,
        "penales_1": int(penales.group(2)) if penales else None,
        "marcador_texto": texto_marcador.strip(),
    }


def buscar_partidos(driver, anio):
    """Todos los partidos de la pagina del mundial (anio), en el orden en que aparecen"""
    driver.get(URL.format(anio=anio))
    partidos = [leer_partido(t) for t in driver.find_elements(By.XPATH, "//table[contains(@class, 'vevent')]")]
    df = pd.DataFrame([p for p in partidos if p])
    df["year"] = anio
    return df


def main():
    parser = argparse.ArgumentParser(description="Descarga los partidos de un mundial desde Wikipedia")
    grupo = parser.add_mutually_exclusive_group(required=True)
    grupo.add_argument("--anio", type=int, choices=AGNOS)
    grupo.add_argument("--todos", action="store_true")
    parser.add_argument("--salida", type=Path, default=SALIDA)
    args = parser.parse_args()

    args.salida.mkdir(parents=True, exist_ok=True)
    driver = crear_driver()
    try:
        for anio in (AGNOS if args.todos else [args.anio]):
            df = buscar_partidos(driver, anio)
            df.to_csv(args.salida / f"partidos_{anio}.csv", index=False, encoding="utf-8")
            print(f"{anio}: {len(df)} partidos ({int(df['penales_0'].notna().sum())} por penales)")
    finally:
        driver.quit()


if __name__ == "__main__":
    main()
