"""
Descarga de Wikipedia (es) las tablas de posiciones de la fase de grupos de un mundial y las guarda en
data/grupos/grupos_mundiales_<año>.csv (columnas: indice = seleccion, Pts, PJ, PG, PE, PP, GF, GC, Dif, Grupo, year).

Reemplaza a los 20 scripts busca_grupos_AAAA.py (uno por mundial, ver historial de git y README). Aquellos usaban selectores distintos por año
(estilo de la tabla, h3 o h4, recortes de tablas) que dejaron de funcionar cuando Wikipedia cambio su HTML. Este
script usa una sola regla para todos los años:

  - tabla de grupo: tabla cuya cabecera tiene la columna "Pts" y que enlaza a selecciones
  - nombre del grupo: el titulo (h2/h3/h4) mas cercano antes de la tabla; si su id no empieza por "Grupo_" la tabla
    no es de grupos (p. ej. "Estadisticas" o la ronda final de 1950) y se descarta

Uso:
    python etl/busca_grupos.py --anio 1986
    python etl/busca_grupos.py --todos --salida data/redescarga/grupos

Requiere Firefox instalado (Selenium descarga geckodriver solo) y requirements-dev.txt.
"""
import argparse
from pathlib import Path

import pandas as pd
from fake_useragent import UserAgent
from selenium.webdriver import Firefox
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options

URL = "https://es.wikipedia.org/wiki/Copa_Mundial_de_F%C3%BAtbol_de_{anio}"
SALIDA = Path(__file__).resolve().parent.parent / "data" / "grupos"
# mundiales con fase de grupos (1934 y 1938 fueron de eliminacion directa)
AGNOS = [1930, 1950, 1954, 1958, 1962, 1966, 1970, 1974, 1978, 1982, 1986, 1990, 1994, 1998, 2002, 2006, 2010, 2014, 2018, 2022]

# Corea del Sur no sigue el patron "Selección de fútbol de ..." en el title de su enlace
ENLACE_EQUIPO = ".//a[contains(@title, 'Selección de fútbol de') or @title='Corea del Sur']"
TABLAS = f"//table[.//tr[1]/th[starts-with(normalize-space(.), 'Pts')] and {ENLACE_EQUIPO[1:]}]"
TITULO_PREVIO = "./preceding::*[self::h2 or self::h3 or self::h4][1]"


def crear_driver(ver_navegador=False, firefox=None):
    """Firefox con un user-agent aleatorio; sin ventana salvo que se pida (ver_navegador)"""
    opciones = Options()
    opciones.set_preference("general.useragent.override", UserAgent().random)
    if not ver_navegador:
        opciones.add_argument("--headless")
    if firefox:
        opciones.binary_location = firefox
    return Firefox(options=opciones)


def leer_fila(tr):
    """
    Valores de una fila: la seleccion (texto de su enlace) seguida del texto de las demas celdas (Pts, PJ, ...).
    Devuelve None si la fila no tiene enlace a una seleccion (filas de separacion o notas)
    """
    celdas = tr.find_elements(By.XPATH, "./td")
    for i, td in enumerate(celdas):
        enlaces = td.find_elements(By.XPATH, ENLACE_EQUIPO)
        if enlaces:
            return [enlaces[0].text.strip()] + [c.text.strip() for c in celdas[i + 1:]]
    return None


def buscar_grupos(driver, anio):
    """
    ¿QUE HACE?
    Lee de la pagina del mundial (anio) todas las tablas de posiciones de la fase de grupos

    ¿COMO LO HACE?
    0-) Abre la pagina y busca las tablas con columna "Pts" que enlazan a selecciones (TABLAS)

    1-) Por cada tabla toma el titulo previo mas cercano; si su id no empieza por "Grupo_" la descarta, si no su texto es el nombre del grupo

    2-) Toma los nombres de columna de la cabecera (las que van despues de la columna de la seleccion) y lee cada fila con "leer_fila"

    Returns:

      - df con indice = seleccion, columnas de la tabla, "Grupo" y "year"
    """
    driver.get(URL.format(anio=anio))
    columnas, filas = None, []

    for tabla in driver.find_elements(By.XPATH, TABLAS):
        titulo = tabla.find_elements(By.XPATH, TITULO_PREVIO)
        if not titulo or not (titulo[0].get_attribute("id") or "").startswith("Grupo_"):
            continue
        grupo = titulo[0].text.strip()

        trs = tabla.find_elements(By.XPATH, ".//tr")
        cabecera = [th.text.strip() for th in trs[0].find_elements(By.XPATH, "./th")]
        for tr in trs[1:]:
            valores = leer_fila(tr)
            if valores:
                filas.append(valores + [grupo])
                if columnas is None:
                    columnas = cabecera[-(len(valores) - 1):]

    if not filas:
        raise RuntimeError(f"No se encontraron tablas de grupos en {URL.format(anio=anio)}")
    df = pd.DataFrame([f[1:] for f in filas], index=[f[0] for f in filas], columns=columnas + ["Grupo"])
    df["year"] = str(anio)
    return df


def main():
    parser = argparse.ArgumentParser(description="Descarga las tablas de la fase de grupos de un mundial desde Wikipedia")
    grupo = parser.add_mutually_exclusive_group(required=True)
    grupo.add_argument("--anio", type=int, choices=AGNOS)
    grupo.add_argument("--todos", action="store_true", help="descarga todos los mundiales con fase de grupos")
    parser.add_argument("--salida", type=Path, default=SALIDA, help="carpeta de los csv (por defecto data/grupos)")
    parser.add_argument("--ver-navegador", action="store_true", help="abre Firefox con ventana en vez de en segundo plano")
    parser.add_argument("--firefox", help="ruta a firefox.exe si no esta en la ubicacion por defecto")
    args = parser.parse_args()

    args.salida.mkdir(parents=True, exist_ok=True)
    driver = crear_driver(args.ver_navegador, args.firefox)
    try:
        for anio in (AGNOS if args.todos else [args.anio]):
            ruta = args.salida / f"grupos_mundiales_{anio}.csv"
            df = buscar_grupos(driver, anio)
            df.to_csv(ruta)
            print(f"{anio}: {len(df)} selecciones -> {ruta}")
    finally:
        driver.quit()


if __name__ == "__main__":
    main()
