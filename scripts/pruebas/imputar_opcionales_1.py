import pandas as pd

BASE_METRICS = None
OPTIONAL_METRICS = None
ALL_METRICS = BASE_METRICS + OPTIONAL_METRICS

def formar_dataset_real():
    pass

def funcion_tabla_desempegno(df0, pais, indx=None, agno=None, ind0=0):
    """
    ¿QUE HACE?
    Calcula y acumula las métricas de desempeño base (BASE_METRICS: PTS, PJ,
    PG, PP, PE, GF, GC, D) y las métricas opcionales (OPTIONAL_METRICS: VI,
    SOT, PKATT, PKATTALLOW) para el equipo (pais), iterando los partidos de
    df0. Puede usarse para generar el rendimiento acumulado en la fase
    eliminatoria previa al mundial (ind0=0), o para ajustar dicho rendimiento
    en df0 tras las predicciones de goles del modelo (ind0=1).

    Las métricas base se calculan con los goles "score_0" y "score_1". Las
    opcionales se imputan a partir del histórico previo de cada equipo,
    guardado en la variable global opcionales. El acumulado corresponde a pais,
    tanto si juega como home como si juega como away.

    ¿COMO LO HACE?
    0-) Declara opcionales como variable global.

    1-) Si opcionales es None:
      1.0-) Carga el histórico con
            formar_dataset_real("proxy_desempegno", agno).

    2-) Crea una copia de opcionales y la indexa por pais para consultar los
        valores por equipo. Si hay varios registros para un país, conserva el
        primero. La función interna valor_previo devuelve el valor de la
        métrica para el equipo o 0.1 si no existe.

    3-) Crea el diccionario acumulador dic_dsp, con una clave por cada métrica
        de ALL_METRICS y un valor inicial de 0.

    4-) Recorre las filas de una copia de df0 como tuplas, conservando el índice
        original en partido.Index.

    5-) Determina en qué lado juega pais:
      5.0-) Si pais está en home, establece sufijo "_0", el rival y los goles
            correspondientes.
      5.1-) Si pais está en away, establece sufijo "_1", el rival y los goles
            correspondientes.
      5.2-) Si pais no participa en el partido, pasa al siguiente.

    6-) Obtiene el PJ previo de pais y del rival con valor_previo.

    7-) Imputa las métricas opcionales, si están en dic_dsp:
      7.0-) Suma 1 a VI si el rival no marcó goles.
      7.1-) Estima SOT usando el promedio histórico de tiros a puerta de pais,
            ajustado según los goles que marcó en el partido.
      7.2-) Estima PKATT a partir del promedio histórico de pais.
      7.3-) Estima PKATTALLOW a partir del promedio histórico de PKATT del rival.

    8-) Actualiza las métricas base según el resultado del partido:
      8.0-) Si pais gana, suma 1 a PG y 3 a PTS.
      8.1-) Si pais pierde, suma 1 a PP.
      8.2-) Si empata, suma 1 a PE y 1 a PTS.
      8.3-) Actualiza GF, GC, D y PJ.

    9-) Si ind0 es 1, actualiza df0 con las métricas acumuladas de pais para
        cada partido:
      9.0-) Escribe las métricas en las columnas correspondientes al lado en
            que jugó pais ("_0" para home o "_1" para away).

    10-) Si ind0 es 0:
      10.0-) Retorna un DataFrame de una fila con year, pais y las métricas
             acumuladas, usando indx como índice.

    Args:
      df0: DataFrame con partidos y las columnas home, away, score_0 y score_1.
           Si ind0=1, se actualiza in-place con las columnas de rendimiento.
      pais: Equipo cuyo rendimiento se calcula o actualiza.
      indx: Índice de la fila retornada cuando ind0=0.
      agno: Año del mundial; también se usa al cargar opcionales.
      ind0: 0 para retornar el acumulado; 1 para actualizar df0 in-place.

    Globales usadas:
      BASE_METRICS: Métricas base.
      OPTIONAL_METRICS: Métricas opcionales.
      ALL_METRICS: Unión de BASE_METRICS y OPTIONAL_METRICS.
      opcionales: DataFrame con el histórico previo de métricas por país.

    Return:
      Si ind0 es 0, retorna un DataFrame de una fila con year, pais y las
      métricas acumuladas, usando indx como índice.
      Si ind0 es 1, retorna None y actualiza df0 in-place.
    """

    # 0-) Declara opcionales como variable global.
    global opcionales

    # 1-) Comprueba si hay que cargar el histórico.
    if opcionales is None:
        # 1.0-) Carga el histórico previo de métricas por país.
        opcionales = formar_dataset_real("proxy_desempegno", agno)

    # 2-) Indexa una copia por país para consultar los valores históricos.
    # Conserva el primer registro si hay países duplicados.
    cop_opcionales = opcionales.drop_duplicates(subset="pais", keep="first").copy()
    cop_opcionales.index = cop_opcionales["pais"]

    def valor_previo(metrica, equipo, defecto=0.1):
        return cop_opcionales[metrica].get(equipo, defecto)

    # 3-) Inicializa el acumulador de métricas.
    dic_dsp = {m: 0 for m in ALL_METRICS}

    # 4-) Recorre una copia de los partidos, conservando sus índices originales.
    for partido in df0.copy().itertuples():
        # 5-) Determina el lado en que juega pais y asigna el rival y los goles.
        if partido.home == pais:
            # 5.0-) pais juega como home.
            sufijo, rival = "_0", partido.away
            gf_pais, gf_rival = partido.score_0, partido.score_1
        elif partido.away == pais:
            # 5.1-) pais juega como away.
            sufijo, rival = "_1", partido.home
            gf_pais, gf_rival = partido.score_1, partido.score_0
        else:
            # 5.2-) pais no participa en este partido.
            continue

        # 6-) Obtiene los partidos previos de pais y del rival.
        pj_previo = valor_previo("PJ", pais)
        pj_previo_rival = valor_previo("PJ", rival)

        # 7-) Imputa las métricas opcionales presentes en el acumulador.
        if "VI" in dic_dsp:
            # 7.0-) Suma una valla invicta si el rival no marcó.
            dic_dsp["VI"] += 1 if gf_rival == 0 else 0

        if "SOT" in dic_dsp:
            # 7.1-) Estima los tiros a puerta a partir del promedio histórico.
            avg_sot = valor_previo("SOT", pais) / max(1, pj_previo)
            ratio = gf_pais / max(0.5, avg_sot * 0.3)
            dic_dsp["SOT"] += round(
                avg_sot * (0.7 + 0.3 * min(ratio, 2)), 1
            )

        if "PKATT" in dic_dsp:
            # 7.2-) Estima los penales intentados por pais.
            dic_dsp["PKATT"] += round(
                valor_previo("PKATT", pais) / max(1, pj_previo) * 0.9, 2
            )

        if "PKATTALLOW" in dic_dsp:
            # 7.3-) Estima los penales permitidos a partir del histórico del rival.
            dic_dsp["PKATTALLOW"] += round(
                valor_previo("PKATT", rival) / max(1, pj_previo_rival) * 0.9, 2
            )

        # 8-) Actualiza las métricas base según el resultado.
        if gf_pais > gf_rival:
            # 8.0-) Victoria.
            dic_dsp["PG"] += 1
            dic_dsp["PTS"] += 3
        elif gf_pais < gf_rival:
            # 8.1-) Derrota.
            dic_dsp["PP"] += 1
        else:
            # 8.2-) Empate.
            dic_dsp["PE"] += 1
            dic_dsp["PTS"] += 1

        # 8.3-) Actualiza goles, diferencia y partidos jugados.
        dic_dsp["GF"] += gf_pais
        dic_dsp["GC"] += gf_rival
        dic_dsp["D"] = dic_dsp["GF"] - dic_dsp["GC"]
        dic_dsp["PJ"] += 1

        # 9-) Si corresponde, actualiza df0 con el acumulado hasta este partido.
        if ind0 == 1:
            # 9.0-) Escribe las métricas en las columnas del lado de pais.
            cols_to_update = [f"{m}{sufijo}" for m in ALL_METRICS]
            df0.loc[partido.Index, cols_to_update] = [
                dic_dsp[m] for m in ALL_METRICS
            ]

    # 10-) Si se solicitó el acumulado para la fase eliminatoria, lo retorna.
    if ind0 == 0:
        # 10.0-) Construye y retorna el DataFrame de una fila.
        return pd.DataFrame(
            {"year": agno, "pais": pais, **dic_dsp},
            index=[indx],
        )