import json

import numpy as np
import pandas as pd

def formar_dataset_real():
    pass

def calc_gkps():
    pass
def calc_mds():
    pass
def calc_mos():
    pass
def calc_mms():
    pass
def calc_rate():
    pass
def calc_rate():
    pass
def obtener_diferencia():
    pass

OPTIONAL_METRICS = None

def calculo_metricas_0(df_desempegno, agno=None):
    """
    ¿QUE HACE?
    Prepara las métricas de desempeño para los equipos home y away de cada
    partido. Calcula métricas especializadas por país a partir de las
    estadísticas del equipo y de sus datos históricos opcionales. Conserva
    todas las filas de df_desempegno para que el resultado siga alineado con
    los partidos originales.

    ¿COMO LO HACE?
    0-) Declara opcionales como variable global.

    1-) Si opcionales es None, carga el histórico con
        formar_dataset_real("proxy_desempegno", agno).

    2-) Obtiene los países presentes en las columnas home y away.

    3-) Selecciona del histórico los datos de esos países. Si hay más de una
        fila histórica para un país, conserva la primera. Para un equipo sin
        datos históricos, usa 0.1 en las métricas opcionales.

    4-) Prepara, por separado, las columnas de rendimiento de home y away.
        Selecciona las métricas por su sufijo (_0 o _1), sin incluir score_0
        ni score_1.

    5-) Asocia a cada equipo sus datos históricos mediante un left merge, de
        modo que ningún partido se descarte por falta de datos opcionales.

    6-) Agrupa los partidos por país y calcula sus métricas especializadas
        mediante calc_gkps, calc_mds, calc_mos, calc_mms, calc_rate y
        obtener_diferencia.

    7-) Elimina de la salida las métricas históricas intermedias
        (VI, SOT, PKATT y PKATTALLOW) y añade las métricas especializadas.

    8-) Asigna 0.1 a las columnas indicadas por DROP_METRICS, si las hay.

    9-) Combina los resultados de home y away manteniendo el orden de las
        filas de entrada.

    10-) Retorna el DataFrame preparado para el modelo.

    Args:
      df_desempegno: DataFrame de partidos con home, away, score_0, score_1
          y las métricas de rendimiento de cada equipo, identificadas por
          los sufijos _0 y _1.
      agno: Año usado para cargar los datos históricos opcionales.

    Globales usadas:
      opcionales: DataFrame histórico con pais, VI, SOT, PKATT y PKATTALLOW.
      DROP_METRICS: Columnas que se reemplazan por el valor 0.1.

    Return:
      DataFrame con las métricas preparadas para los equipos home y away,
      conservando una fila por cada partido de df_desempegno.
    """

    # 0-) Declara opcionales como variable global.
    global opcionales

    # 1-) Carga el histórico cuando todavía no está disponible.
    if opcionales is None:
        opcionales = formar_dataset_real("proxy_desempegno", agno)

    # 2-) Obtiene los países presentes en los partidos.
    paises_unicos = pd.concat(
        [df_desempegno["home"], df_desempegno["away"]],
        ignore_index=True,
    ).unique()

    # 3-) Prepara los datos históricos y conserva como máximo una fila por país.
    columnas_opcionales = ["VI", "SOT", "PKATT", "PKATTALLOW"]
    insumos_apuntados = (
        opcionales.loc[
            opcionales["pais"].isin(paises_unicos),
            ["pais", *columnas_opcionales],
        ]
        .drop_duplicates(subset="pais", keep="first")
    )

    # 4-) Selecciona las métricas de cada lado sin depender del orden de columnas.
    df1 = df_desempegno.copy()

    columnas_home = [
        "home",
        *[
            columna
            for columna in df1.columns
            if columna.endswith("_0") and columna != "score_0"
        ],
    ]
    columnas_away = [
        "away",
        *[
            columna
            for columna in df1.columns
            if columna.endswith("_1") and columna != "score_1"
        ],
    ]

    homes = df1.loc[:, columnas_home].rename(columns={"home": "pais"})
    aways = df1.loc[:, columnas_away].rename(columns={"away": "pais"})

    df_nuevo = pd.DataFrame()
    columnas = []
    equipos = {0: "home", 1: "away"}

    # 5-) Procesa home y away por separado y asocia el histórico sin descartar filas.
    for idx, trab in enumerate([homes, aways]):
        trab = trab.merge(
            insumos_apuntados,
            on="pais",
            how="left",
            sort=False,
        )

        # Los países sin datos históricos conservan sus partidos con este valor
        # de respaldo, en vez de desaparecer por un inner merge.
        trab.loc[:, columnas_opcionales] = trab[columnas_opcionales].fillna(0.1)

        # 6-) Calcula las métricas especializadas agrupando por país.
        df_especializado = (
            trab.groupby("pais", as_index=False)
            .apply(
                lambda g: pd.Series(
                    {
                        f"gkps_{idx}": calc_gkps(
                            g[f"PJ_{idx}"],
                            g[f"GC_{idx}"],
                            g["VI"],
                            g["PKATTALLOW"],
                        ),
                        f"mds_{idx}": calc_mds(
                            g[f"PJ_{idx}"],
                            g[f"GC_{idx}"],
                            g["VI"],
                            g["PKATTALLOW"],
                        ),
                        f"mos_{idx}": calc_mos(
                            g[f"PJ_{idx}"],
                            g[f"GF_{idx}"],
                            g["SOT"],
                            g["PKATT"],
                        ),
                        f"mms_{idx}": calc_mms(
                            g[f"PJ_{idx}"],
                            g[f"PTS_{idx}"],
                            g[f"PG_{idx}"],
                            g["SOT"],
                        ),
                        f"rate_GC_{idx}": calc_rate(
                            g[f"GC_{idx}"],
                            "GC",
                            num=idx,
                        ),
                        f"rate_GF_{idx}": calc_rate(
                            g[f"GF_{idx}"],
                            "GF",
                            num=idx,
                        ),
                        f"D_{idx}": obtener_diferencia(g, num=idx),
                    }
                )
            )
            .reset_index(drop=True)
        )

        # 7-) Quita las métricas intermedias y añade las especializadas.
        trab = trab.drop(columns=columnas_opcionales)
        df_provisional = trab.merge(
            df_especializado,
            on="pais",
            how="left",
            sort=False,
        )
        df_provisional = df_provisional.rename(
            columns={"pais": equipos[idx]}
        )

        # 8-) Reemplaza por 0.1 las métricas indicadas para descartar.
        df_provisional.loc[:, DROP_METRICS] = 0.1

        # 9-) Concatena por posición para mantener una fila por partido.
        df_provisional = df_provisional.reset_index(drop=True)
        df_nuevo = pd.concat(
            [df_nuevo.reset_index(drop=True), df_provisional],
            axis=1,
            ignore_index=True,
        )
        columnas.extend(df_provisional.columns)

    # 10-) Restaura los nombres de las columnas y retorna el resultado.
    df_nuevo.columns = columnas
    return df_nuevo