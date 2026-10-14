import pandas as pd 

OPTIONAL_METRICS, ALL_METRICS = None, None

def formar_dataset_real(tipo, agno):
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
    opcionales se imputan a partir del progreso de cada equipo: el histórico
    previo guardado en la variable global opcionales, avanzado partido a
    partido en una copia local (cop_opcionales) que no altera la global. El
    acumulado corresponde a pais, tanto si juega como home como si juega como
    away. Con ind0=1 retorna además el progreso de pais, para que
    calculo_metricas_1 actualice opcionales tras procesar todos los países.

    ¿COMO LO HACE?
    0-) Declara opcionales como variable global.

    1-) Si hay métricas opcionales (OPTIONAL_METRICS no vacío) y opcionales es
        None:
      1.0-) Carga el histórico con
            formar_dataset_real("proxy_desempegno", agno).

    2-) Si hay métricas opcionales, crea una copia local cop_opcionales de
        opcionales y la indexa por pais. Si hay varios registros para un país,
        conserva el primero. Define dos funciones internas:
      2.0-) valor_previo: devuelve el valor de la métrica para el equipo en
            cop_opcionales, o un valor por defecto (0.1) si el equipo o la
            métrica no existen o el valor es NaN.
      2.1-) avanzar: suma un incremento a la métrica del equipo en
            cop_opcionales, agregando al equipo si no tenía histórico.

    3-) Inicializa el acumulador dic_dsp en None; se llenará en el primer
        partido de pais.

    4-) Recorre las filas de una copia de df0 como tuplas, conservando el índice
        original en partido.Index.

    5-) Determina en qué lado juega pais:
      5.0-) Si pais está en home, establece sufijo "_0", el rival y los goles
            correspondientes.
      5.1-) Si pais está en away, establece sufijo "_1", el rival y los goles
            correspondientes.
      5.2-) Si pais no participa en el partido, pasa al siguiente.
      5.3-) Si es el primer partido de pais, inicializa dic_dsp con los
            valores de la fila para cada métrica con el sufijo de pais (lo
            traído de la fase previa o de eliminatorias), o 0 si df0 no tiene
            esa columna o el valor es NaN.

    6-) Obtiene el PJ previo (del progreso local) de pais y del rival con
        valor_previo.

    7-) Imputa las métricas opcionales con el progreso previo al partido y las
        suma a dic_dsp si están en él:
      7.0-) VI: 1 si el rival no marcó goles, 0 en otro caso.
      7.1-) SOT: promedio de tiros a puerta de pais, ajustado según los goles
            que marcó en el partido.
      7.2-) PKATT: promedio de penales intentados por pais, por 0.9.
      7.3-) PKATTALLOW: promedio de penales intentados por el rival, por 0.9
            (los penales que intenta el rival son los que permite pais).
      7.4-) Avanza el progreso local de pais: suma 1 a PJ y lo imputado a VI,
            SOT, PKATT y PKATTALLOW.
      7.5-) Avanza el progreso local del rival: suma 1 a PJ y a PKATT lo
            imputado como PKATTALLOW de pais, para equilibrar los próximos
            cruces de pais contra ese rival.

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
      10.0-) Si pais no jugó ningún partido, deja dic_dsp en 0 para cada
             métrica.
      10.1-) Retorna un DataFrame de una fila con year, pais y las métricas
             acumuladas, usando indx como índice.

    11-) Si ind0 es 1:
      11.0-) Retorna solo la fila de progreso de pais en cop_opcionales (el
             progreso local del rival se descarta, pues su propia pasada
             calcula el suyo), o None si no hay métricas opcionales.

    Args:
      df0: DataFrame con partidos y las columnas home, away, score_0 y score_1.
           Debe estar en orden cronológico. Si ind0=1, se actualiza in-place
           con las columnas de rendimiento.
      pais: Equipo cuyo rendimiento se calcula o actualiza.
      indx: Índice de la fila retornada cuando ind0=0.
      agno: Año del mundial; también se usa al cargar opcionales.
      ind0: 0 para retornar el acumulado; 1 para actualizar df0 in-place y
            retornar el progreso de pais.

    Globales usadas:
      BASE_METRICS: Métricas base.
      OPTIONAL_METRICS: Métricas opcionales.
      ALL_METRICS: Unión de BASE_METRICS y OPTIONAL_METRICS.
      opcionales: DataFrame con columnas pais, PJ, VI, SOT, PKATT y
                  PKATTALLOW con el progreso previo por país. Solo se lee;
                  la actualiza calculo_metricas_1.

    Return:
      Si ind0 es 0, retorna un DataFrame de una fila con year, pais y las
      métricas acumuladas, usando indx como índice.
      Si ind0 es 1, actualiza df0 in-place y retorna un DataFrame de una fila
      con el progreso de pais (o None si no hay métricas opcionales).
    """

    # 0-) Declara opcionales como variable global.
    global opcionales

    # 1-) Comprueba si hay que cargar el histórico.
    if OPTIONAL_METRICS and opcionales is None:
        # 1.0-) Carga el histórico previo de métricas por país.
        opcionales = formar_dataset_real("proxy_desempegno", agno)

    # 2-) Copia local indexada por país: aquí avanza el progreso sin tocar la global.
    # Conserva el primer registro si hay países duplicados.
    cop_opcionales = None
    if OPTIONAL_METRICS:
        cop_opcionales = opcionales.drop_duplicates(subset="pais", keep="first").copy()
        cop_opcionales.index = cop_opcionales["pais"]

    # 2.0-) Valor del progreso local o el valor por defecto.
    def valor_previo(metrica, equipo, defecto=0.1):
        if cop_opcionales is None or metrica not in cop_opcionales.columns:
            return defecto
        valor = cop_opcionales[metrica].get(equipo, defecto)
        return defecto if pd.isna(valor) else valor

    # 2.1-) Suma lo imputado al progreso local; agrega al equipo si no tenía histórico.
    def avanzar(metrica, equipo, incremento, defecto=0.1):
        cop_opcionales.loc[equipo, "pais"] = equipo
        cop_opcionales.loc[equipo, metrica] = valor_previo(metrica, equipo, defecto) + incremento

    # 3-) El acumulador se inicializa en el primer partido de pais.
    dic_dsp = None

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

        # 5.3-) Primer partido de pais: arranca desde lo que trae la fila, o 0.
        if dic_dsp is None:
            dic_dsp = {}
            for m in ALL_METRICS:
                v = getattr(partido, f"{m}{sufijo}", 0)
                dic_dsp[m] = 0 if pd.isna(v) else v

        # 6-) Obtiene los partidos previos (progreso local) de pais y del rival.
        pj_previo = valor_previo("PJ", pais)
        pj_previo_rival = valor_previo("PJ", rival)

        # 7-) Imputa las métricas opcionales con el progreso previo al partido.
        # 7.0-) Valla invicta si el rival no marcó.
        vi_imp = 1 if gf_rival == 0 else 0
        sot_imp = pkatt_imp = pkatt_rival_imp = 0
        if OPTIONAL_METRICS:
            # 7.1-) Tiros a puerta de pais ajustados por sus goles.
            avg_sot = valor_previo("SOT", pais) / max(1, pj_previo)
            ratio = gf_pais / max(0.5, avg_sot * 0.3)
            sot_imp = round(avg_sot * (0.7 + 0.3 * min(ratio, 2)), 1)
            # 7.2-) Penales intentados por pais.
            pkatt_imp = round(valor_previo("PKATT", pais) / max(1, pj_previo) * 0.9, 2)
            # 7.3-) Penales intentados por el rival = permitidos por pais.
            pkatt_rival_imp = round(valor_previo("PKATT", rival) / max(1, pj_previo_rival) * 0.9, 2)

        # 7.0-7.3-) Suma lo imputado a las métricas presentes en el acumulador.
        if "VI" in dic_dsp:
            dic_dsp["VI"] += vi_imp
        if "SOT" in dic_dsp:
            dic_dsp["SOT"] += sot_imp
        if "PKATT" in dic_dsp:
            dic_dsp["PKATT"] += pkatt_imp
        if "PKATTALLOW" in dic_dsp:
            dic_dsp["PKATTALLOW"] += pkatt_rival_imp

        if OPTIONAL_METRICS:
            # 7.4-) Avanza el progreso local de pais.
            avanzar("PJ", pais, 1, defecto=0)
            avanzar("VI", pais, vi_imp, defecto=0)
            avanzar("SOT", pais, sot_imp)
            avanzar("PKATT", pais, pkatt_imp)
            avanzar("PKATTALLOW", pais, pkatt_rival_imp)

            # 7.5-) Avanza el progreso local del rival para los próximos cruces de pais contra él.
            avanzar("PJ", rival, 1, defecto=0)
            avanzar("PKATT", rival, pkatt_rival_imp)

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
            df0.loc[partido.Index, cols_to_update] = [dic_dsp[m] for m in ALL_METRICS]

    # 10-) Si se solicitó el acumulado para la fase eliminatoria, lo retorna.
    if ind0 == 0:
        # 10.0-) pais sin partidos: acumulado en 0.
        if dic_dsp is None:
            dic_dsp = {m: 0 for m in ALL_METRICS}
        # 10.1-) Construye y retorna el DataFrame de una fila.
        return pd.DataFrame({"year": agno, "pais": pais, **dic_dsp}, index=[indx])

    # 11-) Tras predicción: retorna el progreso de pais para actualizar la global.
    if cop_opcionales is not None and pais in cop_opcionales.index:
        # 11.0-) Solo la fila de pais; el progreso local del rival se descarta.
        return cop_opcionales.loc[[pais]].reset_index(drop=True)
    return None

def calculo_metricas_1(df_desempegno):
  """
  ¿QUE HACE?
  Ajusta, tras alguna predicción de goles del modelo, las métricas de
  rendimiento de df_desempegno para cada país único que juega como home o
  away, llamando a funcion_tabla_desempegno con ind0=1. Luego vuelca en la
  variable global opcionales el progreso de cada país, para que la siguiente
  fase impute a partir de él.

  ¿COMO LO HACE?
  0-) Declara opcionales como variable global.

  1-) Obtiene la lista paises con los equipos únicos de las columnas home y
      away de df_desempegno.

  2-) Recorre paises:
    2.0-) Llama a funcion_tabla_desempegno(df_desempegno, pais, ind0=1), que
          actualiza df_desempegno in-place. Cada país avanza en local desde la
          misma global, sin ver el progreso de los demás.
    2.1-) Si retorna progreso (no None), lo agrega a la lista progresos.

  3-) Si hay progresos, ya procesados todos los países:
    3.0-) Une los progresos en el DataFrame nuevos.
    3.1-) Reemplaza en opcionales las filas de esos países por las de nuevos,
          conservando las de los países que no jugaron.

  4-) Retorna df_desempegno.

  Args:
    df_desempegno: DataFrame con los partidos de alguna fase del mundial y las
                   predicciones de goles del modelo en score_0 y score_1.

  Globales usadas:
    opcionales: DataFrame con el progreso previo de métricas por país; se
                actualiza aquí.

  Return:
    df_desempegno con las métricas de rendimiento actualizadas.
  """
  # 0-) Declara opcionales como variable global.
  global opcionales

  # 1-) Países únicos de la fase.
  paises = pd.concat([df_desempegno.home, df_desempegno.away], axis=0).unique()
  progresos = []

  # 2-) Cada país avanza en local desde la misma global.
  for indip, pais in enumerate(paises):
    # 2.0-) Ajusta df_desempegno in-place y obtiene el progreso de pais.
    progreso = funcion_tabla_desempegno(df_desempegno, pais, ind0=1)
    # 2.1-) Guarda el progreso si existe.
    if progreso is not None:
      progresos.append(progreso)

  # 3-) Ya procesados todos, vuelca el progreso en la global.
  if progresos:
    # 3.0-) Une los progresos.
    nuevos = pd.concat(progresos, ignore_index=True)
    # 3.1-) Reemplaza las filas de los países que jugaron.
    opcionales = pd.concat(
      [nuevos, opcionales[~opcionales["pais"].isin(nuevos["pais"])]],
      ignore_index=True,
    )

  # 4-) Retorna el df actualizado.
  return df_desempegno
