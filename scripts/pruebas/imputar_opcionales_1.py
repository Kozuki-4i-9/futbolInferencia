def funcion_tabla_desempegno(df0, pais, indx=None, agno=None, ind0=0):
    """
    ¿QUE HACE?
    Calcula y acumula las métricas de desempeño base (BASE_METRICS: PTS, PJ, PG, PP, PE, GF, GC, D) y las métricas opcionales (OPTIONAL_METRICS: VI, SOT, PKATT, PKATTALLOW) para el equipo (pais) indicado, iterando los partidos presentes en el df (df0). Puede ser usado para generar el rendimiento acumulado en la fase eliminatoria previa al mundial (ind0=0), o para ajustar dicho rendimiento dentro del df (df0) tras las predicciones de goles del modelo para alguna fase del mundial (ind0=1). Las métricas base se calculan con los goles "score_0" y "score_1" de cada partido, mientras que las opcionales se imputan (estiman) a partir del histórico previo del equipo y de su rival, guardado en la variable global (opcionales). El acumulado es uno solo para el (pais), sin importar si en cada partido jugó como "home" o como "away"

    ¿COMO LO HACE?
    0-) Se declara como global la variable (opcionales)

    1-) Si la variable global (opcionales) es "None"

      1.0-) Se carga (opcionales) con formar_dataset_real("proxy_desempegno", agno), un df con una columna "pais" y el histórico previo de métricas ("SOT", "PKATT", "PJ", ...) de cada pais

    2-) Se define la funcion interna (valor_previo), que recibe una (metrica) y un (equipo) y retorna el valor de esa metrica para ese equipo dentro de (opcionales), o un valor por defecto de 0.1 si el equipo no aparece

    3-) Se crea el diccionario acumulador (dic_dsp), con una clave por cada metrica de ALL_METRICS (sin sufijo) y valor de inicializacion igual a 0

    4-) Se inicia un bucle for con la variable de iteracion (partido), recorriendo como tuplas todas las filas de una copia del parametro (df0), incluyendo su indice real en "partido.Index"

    5-) Se determina el lado en el que juega el (pais) dentro del (partido)

      5.0-) Si el equipo registrado en "home" es igual al parametro (pais): (sufijo) es "_0" - (rival) es el "away" - (gf_pais) es el "score_0" y (gf_rival) es el "score_1"

      5.1-) Si no, si el equipo registrado en "away" es igual al parametro (pais): (sufijo) es "_1" - (rival) es el "home" - (gf_pais) es el "score_1" y (gf_rival) es el "score_0"

      5.2-) Si el (pais) no juega el (partido), se pasa al siguiente

    6-) Con (valor_previo) se obtienen (pj_previo) "PJ" del (pais) y (pj_previo_rival) "PJ" del (rival)

    7-) Se imputan las metricas opcionales, cada una solo si existe en (dic_dsp), es decir, si está en ALL_METRICS

      7.0-) A la clave "VI" "valla invicta" se le suma 1 si (gf_rival) es 0

      7.1-) Se calcula (avg_sot) como el "SOT" previo del (pais) entre max(1, pj_previo) - se calcula (ratio) como (gf_pais) entre max(0.5, avg_sot * 0.3) - a la clave "SOT" se le suma (avg_sot) * (0.7 + 0.3 * min(ratio, 2)) redondeado a 1 decimal

      7.2-) A la clave "PKATT" se le suma el "PKATT" previo del (pais) entre max(1, pj_previo), por 0.9, redondeado a 2 decimales

      7.3-) A la clave "PKATTALLOW" se le suma el "PKATT" previo del (rival) entre max(1, pj_previo_rival), por 0.9, redondeado a 2 decimales

    8-) Se actualizan las metricas base segun el resultado del (partido) visto desde el (pais)

      8.0-) Si (gf_pais) es mayor a (gf_rival): a la clave "PG" se le suma 1 y a la clave "PTS" se le suma 3

      8.1-) Si (gf_pais) es menor a (gf_rival): a la clave "PP" se le suma 1

      8.2-) Si (gf_pais) es igual a (gf_rival): a la clave "PE" se le suma 1 y a la clave "PTS" se le suma 1

      8.3-) A la clave "GF" se le suma (gf_pais) - a la clave "GC" se le suma (gf_rival) - la clave "D" se actualiza como "GF" menos "GC" - a la clave "PJ" se le suma 1

    9-) Si el (ind0) es 1 "ajuste de valores de rendimiento tras prediccion"

      9.0-) El parametro df (df0) se actualiza con un ".loc" en la fila marcada por "partido.Index" y las columnas de ALL_METRICS con el (sufijo) del lado en que jugó el (pais) ("PTS_0", ... o "PTS_1", ...), asignandoles los valores acumulados en (dic_dsp), quedando en esa fila el rendimiento acumulado del (pais) hasta ese partido

    10-) Si (ind0) es 0 "indicando que se forma el rendimiento acumulado en fase eliminatoria"

      10.0-) Se retorna un df de una fila con las columnas "year" y "pais" (iguales a los parametros (agno) y (pais)) seguidas de lo acumulado en (dic_dsp), y con el index igual a lo pasado por el parametro (indx)

    Args:

      - df0: df con partidos de algun mundial o de la fase eliminatoria previa a estos, con columnas "home", "away", "score_0" y "score_1", para entresacar o ajustar el rendimiento. Si ind0=1 se escriben (o crean) en él las columnas de rendimiento con sufijo "_0" y "_1"

      - pais: nombre del pais-equipo a definir o ajustar rendimiento

      - indx: indica el indice que llevará el df retornado para el caso de busqueda de valores de rendimiento en fase eliminatoria

      - agno: el año del mundial trabajado, usado tambien para cargar (opcionales)

      - ind0: si es 0 indica que se trabaja en el caso de generar valores de rendimiento para cada equipo en su fase eliminatoria - si es 1 indica que se ajustarán parametros de rendimiento dentro de (df0) tras prediccion de goles

    Globales usadas:

      - BASE_METRICS: lista de las metricas base (PTS, PJ, PG, PP, PE, GF, GC, D)

      - OPTIONAL_METRICS: lista de metricas opcionales a imputar (VI, SOT, PKATT, PKATTALLOW)

      - ALL_METRICS: union de BASE_METRICS y OPTIONAL_METRICS

      - opcionales: df con columna "pais" y el histórico previo de metricas por pais, se carga perezosamente la primera vez que se necesita

    Return:

      - Si (ind0) es 0: df de una fila con "year", "pais" y lo acumulado en (dic_dsp) (metricas base y opcionales sin sufijo), con el index igual a (indx), para el caso de determinar el rendimiento en fase eliminatoria

      - Si (ind0) es 1: retorna None - solo ajusta in-place dentro de (df0), fila por fila, los valores de rendimiento con sufijo "_0" (si el pais es "home") o "_1" (si el pais es "away") para todas las metricas de ALL_METRICS
    """

    global opcionales

    if opcionales is None:
        opcionales = formar_dataset_real("proxy_desempegno", agno)

    def valor_previo(metrica, equipo, defecto=0.1):
        fila = opcionales.loc[opcionales.pais == equipo, metrica]
        return fila.iloc[0] if not fila.empty else defecto

    dic_dsp = {m: 0 for m in ALL_METRICS}

    for partido in df0.copy().itertuples():
        if partido.home == pais:
            sufijo, rival = "_0", partido.away
            gf_pais, gf_rival = partido.score_0, partido.score_1
        elif partido.away == pais:
            sufijo, rival = "_1", partido.home
            gf_pais, gf_rival = partido.score_1, partido.score_0
        else:
            continue

        pj_previo = valor_previo("PJ", pais)
        pj_previo_rival = valor_previo("PJ", rival)

        if "VI" in dic_dsp:
            dic_dsp["VI"] += 1 if gf_rival == 0 else 0

        if "SOT" in dic_dsp:
            avg_sot = valor_previo("SOT", pais) / max(1, pj_previo)
            ratio = gf_pais / max(0.5, avg_sot * 0.3)
            dic_dsp["SOT"] += round(
                avg_sot * (0.7 + 0.3 * min(ratio, 2)), 1
            )

        if "PKATT" in dic_dsp:
            dic_dsp["PKATT"] += round(
                valor_previo("PKATT", pais) / max(1, pj_previo) * 0.9, 2
            )

        if "PKATTALLOW" in dic_dsp:
            dic_dsp["PKATTALLOW"] += round(
                valor_previo("PKATT", rival) / max(1, pj_previo_rival) * 0.9, 2
            )

        if gf_pais > gf_rival:
            dic_dsp["PG"] += 1
            dic_dsp["PTS"] += 3
        elif gf_pais < gf_rival:
            dic_dsp["PP"] += 1
        else:
            dic_dsp["PE"] += 1
            dic_dsp["PTS"] += 1

        dic_dsp["GF"] += gf_pais
        dic_dsp["GC"] += gf_rival
        dic_dsp["D"] = dic_dsp["GF"] - dic_dsp["GC"]
        dic_dsp["PJ"] += 1

        if ind0 == 1:
            cols_to_update = [f"{m}{sufijo}" for m in ALL_METRICS]
            df0.loc[partido.Index, cols_to_update] = [dic_dsp[m] for m in ALL_METRICS]

    if ind0 == 0:
        return pd.DataFrame({"year": agno, "pais": pais, **dic_dsp}, index=[indx])
