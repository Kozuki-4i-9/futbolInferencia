import pandas as pd

BASE_METRICS, ALL_METRICS, OPTIONAL_METRICS = None, None, None
opcionales = None

def formar_dataset_real(): # PEND recomentar docstring
    pass

def funcion_tabla_desempegno(df0, pais, indx=None, agno=None, ind0=0):
  """
  ¿QUE HACE?
  Calcula y acumula las métricas de desempeño (PTS, PJ, PG, PP, PE, GF, GC, D) para el equipo (pais) indicado, iterando los partidos presentes en el df (df0). Puede ser usado para generar el rendimiento acumulado en la fase eliminatoria previa al mundial (ind0=0), o para ajustar dicho rendimiento en el df (df0) tras las predicciones de goles del modelo para alguna fase del mundial (ind0=1)

  ¿COMO LO HACE?
  0-) Se crea un diccionario (dic_dsp) vacio - Se agregan 2 claves "year, pais" al (dic_dsp) con valores iguales a los parametros (agno) y (pais) respectivamente - Se crea un df vacio (primero) - Se crean los parametros (PTS0, PJ0, PG0, PP0, PE0, GF0, GC0) y (PTS1, PJ1, PG1, PP1, PE1, GF1, GC1) y se les da el valor "None" a todos

  1-) Se inicia un bucle for enumerate con la variable de iteracion (partido) de todas las filas como tupla, de una copia del parametro (df0) que contiene ya sea los valores "home", "score_0", "score_1" y "away" de los partidos de la fase eliminatoria previa al mundial, o los parametros de rendimiento de los equipos "home" y "away" de cierta fase o round del mundial

  2-) Si el equipo registrado en "home" de la variable de iteracion (partido) es igual al parametro (pais)

    2.0-) Si el parametro (ind0) es 0 "generandose datos de rendimiento con partidos de eliminatorias" o 1 "ajustándose datos de rendimiento segun las predicciones para "score_0" y "score_1"", y si "None" esta en alguno de los parametros de rendimiento definidos para el "home" (PTS0, PJ0, PG0, PP0, PE0, GF0, GC0, D0)
    
      2.0.0-) Si el parametro (ind0) es 0

        2.0.0.0-) Se actualiza el valor de los parametros (PTS0, PJ0, PG0, PP0, PE0, GF0, GC0, D0) del "home" a "PTS", "PJ", "PG", "PP", "PE", "GF", "GC", "D"

      2.0.1-) Si el parametro (ind0) es 1

        2.0.1.0-) Se actualiza el valor de los parametros (PTS0, PJ0, PG0, PP0, PE0, GF0, GC0, D0) del "home" a "PTS_0", "PJ_0", "PG_0", "PP_0", "PE_0", "GF_0", "GC_0", "D_0"

    2.1-) Si el df (primero) está vacio

        2.1.0) Se actualiza el (dic_dsp) con o en las claves iguales a los parametros (PTS0, PJ0, PG0, PP0, PE0, GF0, GC0, D0), segun si "ind0" fue igual a 0 o 1 respectivamente, y valores de inicializacion iguales a 0

    2.2-) Si dentro del (partido) iterado, el "score_0" del "home" es mayor al "score_1" del "away"
    
      2.2.0-) En (dic_dsp) a la clave (PG0) se le suma 1 - a la clave (PTS0) se le suma 3 - a la clave (GF0) se le suma el "score_0" del (partido) y a la clave (GC0) se le suma el "score_1" del (partido)

    2.3-) Si dentro del (partido) iterado, el "score_0" del "home" es menor al "score_1" del "away"

      2.3.0-) En (dic_dsp) a la clave (PP0) se le suma 1 - a la clave (GF0) se le suma el "score_0" del (partido) - a la clave (GC0) se le suma "score_1" del (partido)
    
    2.4-) Si dentro del (partido) iterado, el "score_0" del "home" es igual al "score_1" del "away"

      2.4.0-) En (dic_dsp) a la clave (PE0) se le suma 1 - a la clave (PTS0) se le suma 1 - a la clave (GF0) se le suma "score_0" del (partido) - a la clave (GC0) se le suma "score_1" del (partido)

    2.5-) En (dic_dsp) a la clave (D0) la actualizo como la resta entre el valor de la clave (GF0) en (dic_dsp) menos el valor de la clave (GC0) en (dic_dsp) - en (dic_dsp) al valor de la clave (PJ1) se le suma 1 - al df (primero) se le actualiza con el diccionario (dic_dsp)
      
    2.6-) Si el (ind0) es 1 "ajuste de valores de rendimiento tras prediccion"

      2.6.0-) El parametro df (df0) se actualiza con un ".loc" a una serie creada a partir del diccionario (dic_dsp), en la fila marcada por la variable de iteracion (indice) y las columnas marcadas por los parametros (PTS0, PJ0, PG0, PP0, PE0, GF0, GC0, D0)
  
  3-) Si el equipo registrado en "away" de la variable de iteracion (partido) es igual al parametro (pais)

    3.0-) Si el parametro (ind0) es 0 "generandose datos de rendimiento con partidos de eliminatorias" o 1 "ajustándose datos de rendimiento segun las predicciones para "score_0" y "score_1"", y si "None" esta en alguno de los parametros de rendimiento definidos para el "away" (PTS1, PJ1, PG1, PP1, PE1, GF1, GC1, D1)
    
      3.0.0-) Si el parametro (ind0) es 0

        3.0.0.0-) Se actualiza el valor de los parametros (PTS1, PJ1, PG1, PP1, PE1, GF1, GC1, D1) del "away" a "PTS", "PJ", "PG", "PP", "PE", "GF", "GC", "D"

      3.0.1-) Si el parametro (ind0) es 1

        3.0.1.0-) Se actualiza el valor de los parametros (PTS1, PJ1, PG1, PP1, PE1, GF1, GC1, D1) del "home" a "PTS_1", "PJ_1", "PG_1", "PP_1", "PE_1", "GF_1", "GC_1", "D_1"

    3.1-) Si el df (primero) está vacio

        3.1.0) Se actualiza el (dic_dsp) con o en las claves iguales a los parametros (PTS1, PJ1, PG1, PP1, PE1, GF1, GC1, D1), segun si "ind0" fue igual a 0 o 1 respectivamente, y valores de inicializacion iguales a 0

    3.2-) Si dentro del (partido) iterado, el "score_1" del "away" es mayor al "score_0" del "home"
    
      3.2.0-) En (dic_dsp) a la clave (PG1) se le suma 1 - a la clave (PTS1) se le suma 3 - a la clave (GF1) se le suma el "score_1" del (partido) y a la clave (GC1) se le suma el "score_0" del (partido)

    3.3-) Si dentro del (partido) iterado, el "score_1" del "away" es menor al "score_0" del "home"

      3.3.0-) En (dic_dsp) a la clave (PP1) se le suma 1 - a la clave (GF1) se le suma el "score_1" del (partido) - a la clave (GC1) se le suma "score_0" del (partido)
    
    3.4-) Si dentro del (partido) iterado, el "score_1" del "away" es igual al "score_0" del "home"

      3.4.0-) En (dic_dsp) a la clave (PE1) se le suma 1 - a la clave (PTS1) se le suma 1 - a la clave (GF1) se le suma "score_1" del (partido) - a la clave (GC1) se le suma "score_0" del (partido)

    3.5-) En (dic_dsp) a la clave (D1) la actualizo como la resta entre el valor de la clave (GF1) en (dic_dsp) menos el valor de la clave (GC1) en (dic_dsp) - en (dic_dsp) al valor de la clave (PJ1) se le suma 1 - al df (primero) se le actualiza con el diccionario (dic_dsp)
      
    3.6-) Si el (ind0) es 1 "ajuste de valores de rendimiento tras prediccion"

      3.6.0-) El parametro df (df0) se actualiza con un ".loc" a una serie creada a partir del diccionario (dic_dsp), en la fila marcada por la variable de iteracion (indice) y las columnas marcadas por los parametros (PTS1, PJ1, PG1, PP1, PE1, GF1, GC1, D1)
  
  4-) Si (ind0) es 0 "indicando que se forma el rendimiento acumulado en fase eliminatoria"

    4.0-) Se retorna un df hecho a partir de lo acumulado en (dic_dsp) y con el index igual a lo pasado por el parametro (indx)

  Args:

    - df0: df con partidos de algun mundial o fase eliminatoria previa a estos, para ajustar el rendimiento o entresacarlo

    - pais: nombre del pais-equipo a definir o ajustar rendimiento

    - indx: indica el indice que llevará el df retornado para el caso de busqueda de valores de rendimiento en fase eliminatoria

    - agno: el año del mundial trabajado

    - ind0: si es 0 indica que se trabaja en el caso de generar valores de rendimiento para cada equipo en su fase eliminatoria - si es 1 indica que se ajustarán parametros de rendimiento tras prediccion de goles

  Return:

    - df con lo acumulado en (dic_dsp) y con el index igual a (indx), para el caso de determinar el rendimiento en fase eliminatoria, indicado por un (ind0) igual a 0 - solo ajustará tras la inferencia, los valores de rendimiento (PTS0, PJ0, PG0, PP0, PE0, GF0, GC0) o (PTS1, PJ1, PG1, PP1, PE1, GF1, GC1) dentro de (df0), indicado por un (ind0) igual a 1
  """

    global opcionales

    dic_dsp = {}
    dic_dsp["year"] = agno
    dic_dsp["pais"] = pais

    primero = pd.DataFrame()

    suffix_0 = "_0" if ind0 == 1 else ""
    suffix_1 = "_1" if ind0 == 1 else ""

    col_map_0 = {m: f"{m}{suffix_0}" for m in ALL_METRICS}
    col_map_1 = {m: f"{m}{suffix_1}" for m in ALL_METRICS}

    pts_0_col, pj_0_col, pg_0_col, pp_0_col, pe_0_col, gf_0_col, gc_0_col, d_0_col = [col_map_0[m] for m in BASE_METRICS]
    pts_1_col, pj_1_col, pg_1_col, pp_1_col, pe_1_col, gf_1_col, gc_1_col, d_1_col = [col_map_1[m] for m in BASE_METRICS]

    if ind0 == 1 and opcionales is None:
        opcionales = formar_dataset_real("proxy_desempegno", agno)

    for indice, partido in enumerate(df0.copy().itertuples(index=False)):
        if partido.home == pais:
            if primero.empty:
                dic_dsp = {**{col_map_0[m]: 0 for m in BASE_METRICS}, **{col_map_0[m]: 0 for m in OPTIONAL_METRICS}}

            gf_pais = partido.score_0
            gf_rival = partido.score_1

            sot_previo = opcionales["SOT"]; sot_previo.get(pais, 0.1)

            pkatt_previo = opcionales["PKATT"]; pkatt_previo.get(pais, 0.1)
            pkatt_previo_rival = opcionales["PKATT"]; pkatt_previo_rival.get(partido.away, 0.1)                

            pj_previo = opcionales['PJ']; pj_previo.get(pais, 0.1)
            pj_previo_rival = opcionales['PJ']; pj_previo_rival.get(partido.away, 0.1)

            if "VI" in col_map_0:
                dic_dsp[col_map_0["VI"]] += 1 if gf_rival == 0 else 0

            if "SOT" in col_map_0:
                avg_sot = sot_previo / max(1, pj_previo)
                ratio = gf_pais / max(0.5, avg_sot * 0.3)
                dic_dsp[col_map_0["SOT"]] += round(
                    avg_sot * (0.7 + 0.3 * min(ratio, 2)), 1
                )

            if "PKATT" in col_map_0:
                dic_dsp[col_map_0["PKATT"]] += round(
                    pkatt_previo / max(1, pj_previo) * 0.9, 2
                )

            if "PKATTALLOW" in col_map_0:
                dic_dsp[col_map_0["PKATTALLOW"]] += round(
                    pkatt_previo_rival / max(1, pj_previo_rival) * 0.9, 2
                )

            if partido.score_0 > partido.score_1:
                dic_dsp[pg_0_col] += 1
                dic_dsp[pts_0_col] += 3
                dic_dsp[gf_0_col] += partido.score_0
                dic_dsp[gc_0_col] += partido.score_1
            elif partido.score_0 < partido.score_1:
                dic_dsp[pp_0_col] += 1
                dic_dsp[gf_0_col] += partido.score_0
                dic_dsp[gc_0_col] += partido.score_1
            elif partido.score_0 == partido.score_1:
                dic_dsp[pe_0_col] += 1
                dic_dsp[pts_0_col] += 1
                dic_dsp[gf_0_col] += partido.score_0
                dic_dsp[gc_0_col] += partido.score_1

            dic_dsp[d_0_col] = dic_dsp[gf_0_col] - dic_dsp[gc_0_col]
            dic_dsp[pj_0_col] += 1
            primero = pd.DataFrame(data=dic_dsp, index=[0])

            if ind0 == 1:
                cols_to_update = [col_map_0[m] for m in ALL_METRICS]
                df0.loc[indice, cols_to_update] = pd.Series(dic_dsp)

        elif partido.away == pais:
            if primero.empty:
                dic_dsp = {**{col_map_1[m]: 0 for m in BASE_METRICS}, **{col_map_1[m]: 0 for m in OPTIONAL_METRICS}}

            gf_pais = partido.score_1
            gf_rival = partido.score_0

            sot_previo = opcionales["SOT"]; sot_previo.get(pais, 0.1)

            pkatt_previo = opcionales["PKATT"]; pkatt_previo.get(pais, 0.1)
            pkatt_previo_rival = opcionales["PKATT"]; pkatt_previo_rival.get(partido.home, 0.1)                

            pj_previo = opcionales['PJ']; pj_previo.get(pais, 0.1)
            pj_previo_rival = opcionales['PJ']; pj_previo_rival.get(partido.home, 0.1)

            if "VI" in col_map_1:
                dic_dsp[col_map_1["VI"]] += 1 if gf_rival == 0 else 0

            if "SOT" in col_map_1:
                avg_sot = sot_previo / max(1, pj_previo)
                ratio = gf_pais / max(0.5, avg_sot * 0.3)
                dic_dsp[col_map_1["SOT"]] += round(
                    avg_sot * (0.7 + 0.3 * min(ratio, 2)), 1
                )

            if "PKATT" in col_map_1:
                dic_dsp[col_map_1["PKATT"]] += round(
                    pkatt_previo / max(1, pj_previo) * 0.9, 2
                )

            if "PKATTALLOW" in col_map_1:
                dic_dsp[col_map_1["PKATTALLOW"]] += round(
                    pkatt_previo_rival / max(1, pj_previo_rival) * 0.9, 2
                )

            if partido.score_1 > partido.score_0:
                dic_dsp[pg_1_col] += 1
                dic_dsp[pts_1_col] += 3
                dic_dsp[gf_1_col] += partido.score_1
                dic_dsp[gc_1_col] += partido.score_0
            elif partido.score_1 < partido.score_0:
                dic_dsp[pp_1_col] += 1
                dic_dsp[gf_1_col] += partido.score_1
                dic_dsp[gc_1_col] += partido.score_0
            elif partido.score_1 == partido.score_0:
                dic_dsp[pe_1_col] += 1
                dic_dsp[pts_1_col] += 1
                dic_dsp[gf_1_col] += partido.score_1
                dic_dsp[gc_1_col] += partido.score_0

            dic_dsp[d_1_col] = dic_dsp[gf_1_col] - dic_dsp[gc_1_col]
            dic_dsp[pj_1_col] += 1
            primero = pd.DataFrame(data=dic_dsp, index=[0])

            if ind0 == 1:
                cols_to_update = [col_map_1[m] for m in ALL_METRICS]
                df0.loc[indice, cols_to_update] = pd.Series(dic_dsp)

    if ind0 == 0:
        return pd.DataFrame(dic_dsp, index=[indx])

def calculo_metricas_1(df_desempegno):
    global opcionales

    # ... cargar opcionales si hace falta ...

    # PRIMERA PASADA: imputar opcionales dentro de df_desempegno
    # AQUÍ df_desempegno todavía NO tiene los columnas opcionales imputadas
    paises = pd.concat([df_desempegno.home, df_desempegno.away], axis=0).unique()
    for indip, pais in enumerate(paises):
        funcion_tabla_desempegno(df_desempegno, pais, ind0=1)

    # SEGUNDA PASADA: df_desempegno YA pasó por funcion_tabla_desempegno
    # y ya contiene columnas como "VI_0", "SOT_0", ..., "VI_1", "SOT_1", ...
    for _, partido in df_desempegno.iterrows():
        home = partido.home
        away = partido.away

        for metrica in OPTIONAL_METRICS:
            opcionales.loc[opcionales.pais == home, metrica] += partido[f"{metrica}_0"]
            opcionales.loc[opcionales.pais == away, metrica] += partido[f"{metrica}_1"]

    return df_desempegno