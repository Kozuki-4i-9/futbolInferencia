from fake_useragent import UserAgent

from selenium.webdriver import Firefox
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException

import re
import time
import numpy as np
import pandas as pd

# comp 1 = FIFA World Cup en fbref; la url sigue el patron /en/comps/1/{agno}/{seccion}/{agno}-World-Cup-Stats
years = [1982]
dir_gdr = "C:\\Users\\USUARIO\\Trabajo\\proyectos\\mundiales\\futbolInferencia\\data\\proxy_desempegno"

# fbref bloquea por encima de ~10 peticiones/minuto; se espera entre paginas
ESPERA_ENTRE_PAGINAS = 7

# seccion de fbref -> (id de la tabla de equipos, {data-stat: metrica})
FUENTES = {
    "keepers": ("stats_squads_keeper_for", {"gk_clean_sheets": "VI", "gk_pens_att": "PKATTALLOW"}),
    "shooting": ("stats_squads_shooting_for", {"shots_on_target": "SOT", "pens_att": "PKATT"}),
}

# nombres de fbref -> nombres en español usados en la DB (partidos, grupos, proxy_desempegno)
en_to_es = {
    "West Germany": "Alemania", "Germany": "Alemania", "Spain": "España", "Italy": "Italia",
    "Brazil": "Brasil", "France": "Francia", "Netherlands": "Países Bajos", "England": "Inglaterra",
    "Belgium": "Bélgica", "Poland": "Polonia", "Soviet Union": "Unión Soviética",
    "Czechoslovakia": "Checoslovaquia", "Yugoslavia": "Yugoslavia", "Hungary": "Hungría",
    "Austria": "Austria", "Scotland": "Escocia", "Northern Ireland": "Irlanda del Norte",
    "Peru": "Perú", "Chile": "Chile", "Argentina": "Argentina", "Honduras": "Honduras",
    "El Salvador": "El Salvador", "Algeria": "Argelia", "Cameroon": "Camerún", "Kuwait": "Kuwait",
    "New Zealand": "Nueva Zelanda", "Mexico": "México", "Korea Republic": "Corea del Sur",
    "IR Iran": "Irán", "United States": "Estados Unidos", "Republic of Ireland": "Irlanda",
    "Morocco": "Marruecos", "Tunisia": "Túnez", "Saudi Arabia": "Arabia Saudita", "Japan": "Japón",
}


def crear_driver():
    useragent = UserAgent()
    options = Options()
    # 'general.useragent.override' es la preferencia que realmente cambia el user agent en Firefox
    options.set_preference("general.useragent.override", useragent.random)
    options.binary_location = r"C:\\Program Files\\Mozilla Firefox\\firefox.exe"
    service = Service(r"C:\\Users\\USUARIO\\Trabajo\\python\\data\\controladores\\geckodriver.exe")
    driver = Firefox(service=service, options=options)
    driver.implicitly_wait(7)
    driver.maximize_window()
    return driver


def descomentar_tablas(driver):
    """
    fbref envia la mayoria de sus tablas dentro de comentarios html (<!-- ... -->) y las
    muestra por javascript. Si la tabla no aparece en el DOM, se sacan los comentarios
    del div contenedor y se inyectan como html real.
    """
    driver.execute_script("""
        document.querySelectorAll('div.placeholder').forEach(function(p){
            var padre = p.parentNode;
            padre.childNodes.forEach(function(n){
                if (n.nodeType === 8) { padre.insertAdjacentHTML('beforeend', n.nodeValue); }
            });
        });
    """)


def leer_tabla_equipos(driver, id_tabla, columnas):
    """
    ¿QUE HACE?
    Lee la tabla de equipos (id_tabla) de la pagina abierta en (driver) y retorna un df
    con la columna "pais" y una columna por cada metrica de (columnas).

    ¿COMO LO HACE?
    0-) Espera la tabla; si no aparece, descomenta las tablas ocultas y la busca de nuevo.
    1-) Recorre las filas del tbody, ignorando las de cabecera repetida (class 'thead').
    2-) Toma el nombre del equipo del <a> de la celda 'team' (o 'squad' en versiones viejas de fbref).
    3-) Por cada data-stat de (columnas) toma el texto de la celda; si no existe o esta vacia, pone NaN.
    """
    try:
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, id_tabla)))
    except TimeoutException:
        descomentar_tablas(driver)

    try:
        tabla = driver.find_element(By.XPATH, f"//table[@id='{id_tabla}']")
    except NoSuchElementException:
        print(f"No se encontró la tabla {id_tabla}")
        return pd.DataFrame(columns=["pais"] + list(columnas.values()))

    filas = tabla.find_elements(By.XPATH, ".//tbody//tr[not(contains(@class, 'thead'))]")
    registros = []

    for des0 in filas:
        equipo = des0.find_elements(By.XPATH, ".//*[@data-stat='team' or @data-stat='squad']//a")
        if not equipo:
            continue
        registro = {"pais": equipo[0].text.strip()}

        for des1, des2 in columnas.items():
            celda = des0.find_elements(By.XPATH, f".//td[@data-stat='{des1}']")
            texto = celda[0].text.strip().replace(",", "") if celda else ""
            registro[des2] = float(texto) if re.fullmatch(r"-?\d+(\.\d+)?", texto) else np.nan

        registros.append(registro)

    return pd.DataFrame(registros)


def get_metricas_abstractas(year):
    """
    ¿QUE HACE?
    Obtiene VI, SOT, PKATT y PKATTALLOW por seleccion para el mundial (year) desde fbref
    y las guarda en un csv con la forma de la tabla proxy_desempegno.

    ¿COMO LO HACE?
    0-) Abre una pagina por cada seccion de FUENTES (keepers y shooting) y lee su tabla de equipos.
    1-) Une los df por "pais" (outer, para no perder selecciones que falten en una de las paginas).
    2-) Traduce los nombres al español con en_to_es y avisa de los que no tienen traduccion.
    3-) Agrega la columna "year" y guarda el csv.
    """
    driver = crear_driver()
    df_metricas = None

    try:
        for des0, (des1, des2) in FUENTES.items():
            web = f"https://fbref.com/en/comps/1/{year}/{des0}/{year}-World-Cup-Stats"
            driver.get(web)
            df_seccion = leer_tabla_equipos(driver, des1, des2)
            print(f"{year} {des0}: {len(df_seccion)} selecciones")

            df_metricas = df_seccion if df_metricas is None else df_metricas.merge(df_seccion, on="pais", how="outer")
            time.sleep(ESPERA_ENTRE_PAGINAS)
    finally:
        driver.quit()

    no_mapeados = sorted(set(df_metricas["pais"]) - set(en_to_es))
    if no_mapeados:
        print(f"¡Atención! Selecciones sin traducción en {year}: {no_mapeados}")
    df_metricas["pais"] = df_metricas["pais"].map(lambda x: en_to_es.get(x, x))

    df_metricas = df_metricas.reindex(columns=["pais", "VI", "SOT", "PKATT", "PKATTALLOW"])
    df_metricas["year"] = year
    df_metricas.to_csv(dir_gdr + f"\\metricas_abstractas_{year}.csv", index=False, encoding="utf-8")

    return df_metricas


for year in years:
    df_metricas_abstractas = get_metricas_abstractas(year)
    print(df_metricas_abstractas)
