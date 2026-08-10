import numpy as np
import pandas as pd
import tensorflow as tf

from scipy.stats import poisson
from decimal import Decimal
from .models import partidos, fixtures, grupos, clasificaciones

from numpy.random import seed
seed(1)
tf.random.set_seed(2)

agnos = ["1934", "1938", "1950", "1954", "1958", "1962", "1966", "1970", "1974", "1978", "1982", "1986", "1990", "1994", "1998", "2002", "2006", "2010", "2014", "2018", "2026", "2022"]

# Metrics configuration for easy extensibility
BASE_METRICS = ["PTS", "PJ", "PG", "PP", "PE", "GF", "GC", "D"]
OPTIONAL_METRICS = []  # Users can add optional metrics here (e.g., 'fifa_rank')
ALL_METRICS = BASE_METRICS + OPTIONAL_METRICS

# Deep learning parameter configurations
C_S_DELETE_INDEX = 8
C_S_Y_INDEX = 8
N_STEPS = 0

# Slicing/indexing parameters for match features in intsc_prob_goles
MATCH_HOME_IDX = 0
MATCH_AWAY_IDX = 1  # Note: in modulo_8.py, away was fetched from index 1 due to feature zipping.

def formar_dataset_real(ind0, valores=None):
  """
  ¿QUE HACE?
  Busca en la base de datos segun la tabla indicada por "ind0" y segun el año presente en valores (no lo estará en el caso de ind0="clasificaciones").
  
  ¿COMO LO HACE?
  0-) Se tienen 4 bloques condicionales "if", "elif" para cada una de las tablas posibles indicadas por el parametro (ind0): "partidos", "fixtures", "grupos" y "clasificaciones"
    
    0.0-) Si (ind0) es "partidos": 
      
      0.0.0-) Se crea una lista vacia (df0)
      
      0.0.1-) Se itera sobre todos los objetos de la tabla "partidos" de la base de datos, accedida mediante el ORM de Django

      0.0.2-) Si el atributo "agno" del objeto iterado (des) es igual al parametro (valores) (se asume que es un año) y (valores) es diferente a None:

        0.0.2.0-) Se agrega a la lista (df0) una sublista con los atributos "home", "score", "away" y "agno" recuperados del objeto iterado (des)

      0.0.3-) Si el parametro (valores) es igual a None:

        0.0.3.0-) Se agrega a la lista (df0) una sublista con los atributos "home", "score", "away" y "agno" recuperados del objeto iterado (des). Esto ultimo deberia agrgar todas las filas de la tabla "partidos" a (df0)

      0.0.4-) Se crea un df (df) a partir un array con lista de sublistas basada en partidos del año indicado en la variable (valores) con columnas "home", "score", "away" y "agno"

      0.0.5-) Se convierten los tipos de las columnas del df (df) a los tipos adecuados: "O" para cadenas y "int64" para enteros
      0.0.6-) Se retorna el df (df)
    
    0.1-) Si (ind0) es "fixtures":
      0.1.0-) Se crea una lista vacia (df0)
      
      0.1.1-) Se itera sobre todos los objetos de la tabla "fixtures" de la base de datos, accedida mediante el ORM de Django

      0.1.2-) Si el atributo "agno" del objeto iterado (des) es igual al parametro (valores) (se asume que es un año):

        0.1.2.0-) Se agrega a la lista (df0) una sublista con los atributos "home", "score", "away" y "agno" recuperados del objeto iterado (des)

      0.1.3-) Se crea un df (df) a partir un array con la lista de sublistas basada en fixtures del año indicado en la variable (valores) con columnas "home", "score", "away" y "year"

      0.1.4-) Se convierten los tipos de las columnas del df (df) a los tipos adecuados: "O" para cadenas y "int64" para enteros

      0.1.5-) Se retorna el df (df)
    
    0.2-) Si (ind0) es "grupos":
      0.2.0-) Se crea una lista vacia (df0)
      
      0.2.1-) Se itera sobre todos los objetos de la tabla "grupos" de la base de datos, accedida mediante el ORM de Django

      0.2.2-) Si el atributo "agno" del objeto iterado (des) es igual al parametro (valores) (se asume que es un año):
        0.2.2.0-) Se agrega a la lista (df0) una sublista con los atributos "pais", "Pts", "PJ", "PG", "PP", "PE", "GF", "GC", "Dif", "Grupo" y "agno" recuperados del objeto iterado (des)
      
      0.2.3-) Se crea un df (df) a partir un array con la lista de sublistas basada en grupos del año indicado en la variable (valores) con columnas "pais", "Pts", "PJ", "PG", "PP", "PE", "GF", "GC", "Dif", "Grupo" y "agno"

      0.2.4-) Se convierten los tipos de las columnas del df (df) a los tipos adecuados: "O" para cadenas y "int64" para enteros

      0.2.5-) Se retorna el df (df)

    0.3-) Si (ind0) es "clasificaciones":

      0.3.0-) Se crea una lista vacia (df0)
      
      0.3.1-) Se itera sobre todos los objetos de la tabla "clasificaciones" de la base de datos, accedida mediante el ORM de Django

      0.3.2-) Se agrega a la lista (df0) una sublista con los atributos "home", "score_0", "score_1", "away", "tournament" y "agno" recuperados del objeto iterado (des)

      0.3.3-) Se crea un df (df) a partir un array con la lista de sublistas basada en clasificaciones con columnas "home", "score_0", "score_1", "away", "tournament" y "agno"  

      0.3.4-) Se convierten los tipos de las columnas del df (df) a los tipos adecuados: "O" para cadenas y "int64" para enteros

      0.3.5-) Se retorna el df (df)
      
  Args:
  
  - ind0: cadena que indica a que tabla se hará la búsqueda

  - valores: año en que se hará la búsqueda (a no ser que sea un ind0="clasificaciones")

  Returns:

  - df: dataframe con el contenido de la tabla que se ha consultado
  
  """
  if ind0 == "partidos":
    df0 = list()
    for des in partidos.objects.all():
      if des.agno == valores and valores is not None:
        df0.append([des.home, des.score, des.away, des.agno])
      elif valores is None:
        df0.append([des.home, des.score, des.away, des.agno])
    df = pd.DataFrame(data=np.array(df0), columns=["home", "score", "away", "agno"])
    df.home, df.score, df.away, df.agno = df.home.astype('O'), df.score.astype('O'), df.away.astype('O'), df.agno.astype('int64')
    return df

  elif ind0 == "fixtures":
    df0 = list()
    for des in fixtures.objects.all():
      if des.agno == valores:
        df0.append([des.home, des.score, des.away, des.agno])
    df = pd.DataFrame(data=np.array(df0), columns=["home", "score", "away", "year"])
    df.home, df.score, df.away, df.year = df.home.astype('O'), df.score.astype('O'), df.away.astype('O'), df.year.astype('int64')
    return df

  elif ind0 == "grupos":
    df0 = list()
    for des in grupos.objects.all():
      if des.agno == valores:
        df0.append([des.pais, des.Pts, des.PJ, des.PG, des.PP, des.PE, des.GF, des.GC, des.Dif, des.Grupo, des.agno])
    df = pd.DataFrame(data=np.array(df0), columns=["pais", "Pts", "PJ", "PG", "PP", "PE", "GF", "GC", "Dif", "Grupo", "agno"])
    df.pais, df.Pts, df.PJ, df.PG, df.PP, df.PE, df.GF, df.GC, df.Dif, df.Grupo, df.agno = df.pais.astype('O'), df.Pts.astype('int64'), df.PJ.astype('int64'), df.PG.astype('int64'), df.PP.astype('int64'), df.PE.astype('int64'), df.GF.astype('int64'), df.GC.astype('int64'), df.Dif.astype('O'), df.Grupo.astype('O'), df.agno.astype('int64')
    return df

  elif ind0 == "clasificaciones":
    df0 = list()
    for des in clasificaciones.objects.all():
      df0.append([des.home, des.score_0, des.score_1, des.away, des.tournament, des.agno])
    df = np.array(df0)
    df = pd.DataFrame(data=df, columns=["home", "score_0", "score_1", "away", "tournament", "agno"])
    df.home, df.score_0, df.score_1, df.away, df.tournament, df.agno = df.home.astype('O'), df.score_0.astype('int64'), df.score_1.astype('int64'), df.away.astype('O'), df.tournament.astype('O'), df.agno.astype('int64')
    return df

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

    - ind0: si es 0 indica que se trabaja con en el caso de generar valores de rendimiento para cada equipo en su fase eliminatoria - si es 1 indica que se ajustarán parametros de rendimiento tras prediccion de goles

  Return:

    - df con lo acumulado en (dic_dsp) y con el index igual a (indx), para el caso de determinar el rendimiento en fase eliminatoria, indicado por un (ind0) igual a 0 - solo ajustará tras la inferencia, los valores de rendimiento (PTS0, PJ0, PG0, PP0, PE0, GF0, GC0) o (PTS1, PJ1, PG1, PP1, PE1, GF1, GC1) dentro de (df0), indicado por un (ind0) igual a 1
  """
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

  for indice, partido in enumerate(df0.copy().itertuples(index=False)):
    if partido.home == pais:
      if primero.empty:
        init_vals = {col_map_0[m]: 0 for m in BASE_METRICS}
        for m in OPTIONAL_METRICS:
          init_vals[col_map_0[m]] = 0
        dic_dsp.update(init_vals)

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
        init_vals = {col_map_1[m]: 0 for m in BASE_METRICS}
        for m in OPTIONAL_METRICS:
          init_vals[col_map_1[m]] = 0
        dic_dsp.update(init_vals)

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

def calculo_metricas_0(df_desempegno):
  """
  ¿QUE HACE?
  Toma el df de desempeño (df_desempegno), que tiene el orden de juego con el nombre de los equipos y parametros de rendimiento de cada equipo que juega la fase de grupos, o en alguno de los rounds de la fase final, y le agrega 4 columnas nuevas que son "ts_GF_0", "ts_GC_0", "ts_GF_1", "ts_GC_1", que contienen los valores de tasa de goles a favor y en contra para los equipos que juegan como "home" y "away" respectivamente, calculados segun la formula: (goles a favor o en contra / partidos jugados) * 10
  
  ¿COMO LO HACE?
  0-) Se crea un df vacio (df_nw_desempegno) 
  
  1-) Se crean 2 listas (col0_, col1_) contenedoras de los nombres de columnas del df (df_desempegno): (col0_) tomando las primeras 10 columnas mediante slicing (iloc[:,:10]) y descartando con drop las columnas "PJ_0", "GF_0", "GC_0", "D_0"; y (col1_) tomando las columnas desde la posicion 10 en adelante (iloc[:,10:]) y descartando con drop las columnas "PJ_1", "GF_1", "GC_1", "D_1"

    -> quedando algo como:

    col0_ = ["home"   "PTS_0"   "PG_0"   "PP_0"   "PE_0"   "score_0"]   

    col1_ = ["away"   "PTS_1"   "PG_1"   "PP_1"   "PE_1"   "score_1"]   

  2-) Se itera sobre las filas del df (df_desempegno) en forma de tupla
    
    2.0) se coloca todo el contenido de cada linea iterada de la variable (linea) en la nueva variable (lista) 
    
    2.1) se reemplaza esta variable a una nueva donde se eliminan los elementos innecesarios de la lista (en los indices 3, 7, 8, 9, 13, 17, 18, 19) que corresponden a "PJ_0" , "GF_0", "GC_0", "D_0" y "PJ_1", "GF_1", "GC_1", "D_1" respectivamente
    
      -> quedando algo como: 
      
      lista = ["home"   "PTS_0"   "PG_0"   "PP_0"   "PE_0"   "score_0"   "away"   "PTS_1"   "PG_1"   "PP_1"   "PE_1"   "score_1"]   (claves alusivas a los valores que tendrá la lista)
  
    2.2) Se calculan las tasas de goles a favor y en contra para los equipos "home" y "away" (ts_home_GF, ts_away_GF, ts_home_GC, ts_away_GC) segun la formula: (goles a favor o en contra / partidos jugados) * 10, teniendo en cuenta que si los partidos jugados son 0, se reemplaza este valor por 0.085 para evitar division por cero

    2.1) Se crea un diccionario (dic_paises_fg) de relleno para un df (linea_L) con las claves-valor correspondientes a los equipos "home" y "away", los parametros de rendimiento iterados segun las variables (col0_) (col1_) y (lista), mas las nuevas columnas de tasas de goles a favor y en contra y se agrega el parametro de diferencia de goles (D_0, D_1) 
    
    2.2) Se crea un df (linea_L) con el diccionario (dic_paises_fg) and el indice igual al indice de la variable de iteracion (linea.Index)

    2.3) Se concatena el df (linea_L) al df (df_nw_desempegno) a lo largo del eje 0

  Args:
  
  - df_desempegno: df contenedor de los partidos jugados en la fase eliminatoria previa al mundial, o en alguna fase o round del mundial, con las predicciones de goles hechas por el modelo para cada partido

    -> siendo de la forma:

    df_desempegno = ["id0"   "home"   "PTS_0"   "PJ_0"   "PG_0"   "PP_0"   "PE_0"   "GF_0"   "GC_0"   "D_0"   "score_0"   "away"   "PTS_1"   "PJ_1"   "PG_1"   "PP_1"   "PE_1"   "GF_1"   "GC_1"   "D_1"   "score_1"] (claves alusivas a los valores que tendrá la lista)

  Returns:

  - df_nw_desempegno: df contenedor de los partidos jugados en la fase eliminatoria previa al mundial, o en alguna fase o round del mundial, con las predicciones de goles hechas por el modelo para cada partido, mas las nuevas columnas de tasas de goles a favor y en contra para los equipos "home" y "away"

    -> siendo de la forma:

    df_nw_desempegno = ["home"   "PTS_0"   "PG_0"   "PP_0"   "PE_0"   "ts_GF_0"   "ts_GC_0"   "D_0"   "away"   "PTS_1"   "PG_1"   "PP_1"   "PE_1"   "ts_GF_1"   "ts_GC_1"   "D_1"] (claves alusivas a los valores que tendrá la lista)

  """
  df_nw_desempegno = pd.DataFrame()
  
  dropped_metrics = ["PJ", "GF", "GC", "D"]
  keep_metrics = [m for m in ALL_METRICS if m not in dropped_metrics]
  
  col0_ = ["home"] + [f"{m}_0" for m in keep_metrics] + ["score_0"]
  col1_ = ["away"] + [f"{m}_1" for m in keep_metrics] + ["score_1"]

  for linea in df_desempegno.itertuples():
    val0 = [getattr(linea, col) for col in col0_]
    val1 = [getattr(linea, col) for col in col1_]

    ts_home_GF = (linea.GF_0 / (linea.PJ_0 if linea.PJ_0 > 0 else 0.085)) * 10
    ts_away_GF = (linea.GF_1 / (linea.PJ_1 if linea.PJ_1 > 0 else 0.085)) * 10
    ts_home_GC = (linea.GC_0 / (linea.PJ_0 if linea.PJ_0 > 0 else 0.085)) * 10
    ts_away_GC = (linea.GC_1 / (linea.PJ_1 if linea.PJ_1 > 0 else 0.085)) * 10

    dic_paises_fg = {
        **{col: val for col, val in zip(col0_, val0)},
        "ts_GF_0": ts_home_GF, "ts_GC_0": ts_home_GC,
        "D_0": getattr(linea, "D_0", 0),
        **{col: val for col, val in zip(col1_, val1)},
        "ts_GF_1": ts_away_GF, "ts_GC_1": ts_away_GC,
        "D_1": getattr(linea, "D_1", 0)
    }

    linea_L = pd.DataFrame(dic_paises_fg, index=[linea.Index])
    df_nw_desempegno = pd.concat([df_nw_desempegno, linea_L], axis=0)

  return df_nw_desempegno

def calculo_metricas_1(df_desempegno):
  """
  ¿QUE HACE?
  Toma y modifica el df de desempeño (df_desempegno) para cada unico país, indicado en la lista (paises) que sale de la contatenacion a las columnas del df que contienen los posibles equipos que juegaran como "home" y "away", tras alguna predicción. Para esto se llama a la función (funcion_tabla_desempegno) que ajusta los valores de rendimiento (PTS, PJ, PG, PP, PE, GF, GC, D) segun las predicciones de goles hechas por el modelo, para el pais iterado de la lista mencionada
  
  ¿COMO LO HACE?
  0-) Se crea una variable (paises) que contiene la lista de paises unicos a los que se actualizarán, en el df (df_desempegno), los parámetros de desempeño tras la culminación de cada fase o round del mundial. se obtienen los equipos jugadores como "home" y "away" y se concatenan respectivamente en filas dentro de un dataframe, para luego obtener los valores unicos de ambos conjuntos 

  1-) Se itera sobre la variable (paises) con enumerate, obteniendo el indice (indip) y el pais (pais) a actualizar
    
    1.0-) Se llama a la función (funcion_tabla_desempegno) con los parámetros (df_desempegno, pais, ind0=1) para actualizar los valores de desempeño del pais iterado, en este caso, tras la predicción de goles hecha por el modelo como indica el parámetro (ind0=1)
  
  Args:
  
  - df_desempegno: df contenedor de los partidos jugados en la fase eliminatoria previa al mundial, o en alguna fase o round del mundial, con las predicciones de goles hechas por el modelo para cada partido

  Returns:
  
  - df_desempegno: df contenedor de los partidos jugados en la fase eliminatoria previa al mundial, o en alguna fase o round del mundial, con los valores de desempeño actualizados segun la logica presente en la funcion (funcion_tabla_desempegno)

  """
  paises = pd.concat([df_desempegno.home, df_desempegno.away], axis=0).unique()
  for indip, pais in enumerate(paises):
    funcion_tabla_desempegno(df_desempegno, pais, ind0=1)
  return df_desempegno

def agregar_features(df_mundial, dic_ag, ind=None):
  """
  ¿QUE HACE?
  Arroja un df mas completo a partir de uno de entrada que es el parámetro (df_mundial), conteniendo, además de los nombres de equipos home y away, los puntajes para cada uno por partido y el año en que se disputó el mundial, las diferentes métricas de rendimiento que mostrarán los equipos para la inferencia con el modelo, que serán (PTS, PJ, PG, PP, PE, GF, GC, D)
  
  ¿COMO LO HACE?
  0-) Se define una variable (df5) que sera un df vacio. Se crea una variable (llaves) con la lista de llaves del parametro (dic_ag). Se crean unas variables (c_locales, c_visitantes) contenedoras de los nombres de las columnas correspondientes a las metricas de rendimiento para los equipos home y away

  1-) Se itera a la inversa sobre las filas en forma de tupla y con los indices reseteados del df (df_mundial). Se tienen 2 condicionales

    1.0) Si el tipo del año iterado de las filas del df (df_mundial) no es str. Se guarda en las variables (agno0, agno1) el año del mundial en estudio para iniciar la busqueda de informacion de rendimiento en años de clasificacion, casteando el contenido de la columna "agno" a str

    1.1) Si es tipo str el tipo del año iterado de las filas del df (df_mundial). Se guarda en las variables (agno0, agno1) el año del mundial en estudio para iniciar la busqueda de informacion de rendimiento en años de clasificacion en el tipo en que viene
  
  2-) Se guarda en la variable (mundial_actual) la informacion de rendimiento en fase de clasificacion. Se guarda en unas variables (prt_nv_h, prt_nv_a) la informacion de rendimiento registrada para los equipos que juegan como home y away. Se actualizan las variables (agno0, agno1) al indice en la lista (llaves) del año del mundial actual -1.
  
  3-) Se inicia un bucle while en caso de que (prt_nv_h) este vacio no habiendo informacion de rendimiento para el año actual en fase de clasificacion

    3.0) Se actualiza la variable (mundial_actual) con la informacion de rendimiento en clasificaciones del año anterior al del mundial actual. Se actualiza la variable (prt_nv_h) segun el nuevo contenido de (mundial_actual). Se tienen 3 condicionales

      3.0.0) Si (prt_nv_h) no esta vacio. Se rompe el bucle while con break

      3.0.1) Si el indice en (agno0) es 0 y (prt_nv_h) está vacio. Se actualiza (prt_nv_h) con todas las metricas de rendimiento seteadas a 0, con las columnas correspondientes al equipo home

      3.0.2) Si el indice en (agno0) no es 0 y (prt_nv_h) está vacio. Se resta 1 al (agno0)
  
  4-) Se inicia un bucle while en caso de que (prt_nv_a) este vacio no habiendo informacion de rendimiento para el año actual en fase de clasificacion

    4.0) Se actualiza la variable (mundial_actual) con la informacion de rendimiento en clasificaciones del año anterior al del mundial actual. Se actualiza la variable (prt_nv_a) segun el nuevo contenido de (mundial_actual). Se tienen 3 condicionales

      4.0.0) Si (prt_nv_a) no esta vacio. Se rompe el bucle while con break

      4.0.1) Si el indice en (agno0) es 0 y (prt_nv_a) está vacio. Se actualiza (prt_nv_a) con todas las metricas de rendimiento seteadas a 0, con las columnas correspondientes al equipo away

      4.0.2) Si el indice en (agno0) no es 0 y (prt_nv_a) está vacio. Se resta 1 al (agno0)
      
  5-) Se crean 2 condicionales

    5.0) Si el parámetro (ind) es None. Se coloca en un diccionario (data_metricas) toda la informacion de rendimiento obtenida previamente, con claves-columna:
    
    ("home, PTS_0, PJ_0, PG_0, PP_0, PE_0, GF_0, GC_0, D_0, score_0, away, PTS_1, PJ_1, PG_1, PP_1, PE_1, GF_1, GC_1, D_1, score_1")
    
    y valor, la informacion iterada y la registrada en (prt_nv_h) y (prt_nv_a)
    
    5.1) Si el parámetro (ind) es 0. Se coloca en un diccionario (data_metricas) toda la informacion de rendimiento obtenida previamente, con claves-columna:
    
    ("home, PTS_0, PJ_0, PG_0, PP_0, PE_0, GF_0, GC_0, D_0, score_0, away, PTS_1, PJ_1, PG_1, PP_1, PE_1, GF_1, GC_1, D_1, score_1")
    
    Este condicional se encarga de los df fixture, por lo que para los años "1934, 1938" se empieza de una con "score" porque, al no haber fase de grupos (en el fixture) se empieza directamente con la columna score en formato de referencia a partidos (Match X) teniendose unas clave-columna:

    ("home, PTS_0, PJ_0, PG_0, PP_0, PE_0, GF_0, GC_0, D_0, score, away, PTS_1, PJ_1, PG_1, PP_1, PE_1, GF_1, GC_1, D_1, score")

  6-) Se guarda en el df (df5) el contenido de (agf_df)

  7-) Se retorna el contenido de (df5) con filas invertidas e indices reseteados

  Args:

    - df_mundial: df contenedor de los partidos del mundial actual, para obtener las metricas de rendimiento en la fase de clasificaciones

    - dic_ag: diccionario contenedor de df con los parametros de rendimiento (PTS, PJ, PG, PP, PE, GF, GC, D) and the nombre del pais (pais), obtenidos por los equipos durante la fase de clasificacion previa a cada mundial, con clave igual al año del mundial de interés (en formato string) y valor igual al df de rendimiento por equipo
    
    - ind: indicador de si el df (df_mundial) es de partidos historicos o de fixture del mundial en estudio
  
  Returns:

    - df5: df con la informacion del rendimiento de los equipos home y away que jugarán la fase de grupos del mundial actual, durante la fase de clasificacion previa al mundial. Tiene los indices reseteados
  """
  df5 = pd.DataFrame()
  llaves = list(dic_ag.keys())
  c_locales = [f"{m}_0" for m in ALL_METRICS]
  c_visitantes = [f"{m}_1" for m in ALL_METRICS]

  for mnd in (df_mundial[::-1].reset_index(drop=True).itertuples(index=True)):
    if not isinstance(mnd.agno, str):
      agno0 = str(mnd.agno)
      agno1 = str(mnd.agno)
    else:
      agno0 = mnd.agno
      agno1 = mnd.agno

    mundial_actual = dic_ag[agno0]

    prt_nv_h = mundial_actual[mundial_actual.pais==mnd.home][mundial_actual.columns[1:-1]].copy()
    prt_nv_a = mundial_actual[mundial_actual.pais==mnd.away][mundial_actual.columns[1:-1]].copy()
    
    agno0 = llaves.index(agno0) - 1
    agno1 = llaves.index(agno1) - 1

    while prt_nv_h.empty:
      mundial_actual = dic_ag[llaves[agno0]]
      prt_nv_h = mundial_actual[mundial_actual.pais==mnd.home][mundial_actual.columns[1:-1]].copy()

      if not prt_nv_h.empty:
        break

      if agno0 == 0 and prt_nv_h.empty:
        prt_nv_h = pd.DataFrame(np.zeros((1, len(c_locales))), columns=c_locales)
      elif agno0 > 0 and prt_nv_h.empty:
        agno0 -= 1

    while prt_nv_a.empty:
      mundial_actual = dic_ag[llaves[agno1]]
      prt_nv_a = mundial_actual[mundial_actual.pais==mnd.away][mundial_actual.columns[1:-1]].copy()

      if not prt_nv_a.empty:
        break
      
      if agno1 == 0 and prt_nv_a.empty:
        prt_nv_a = pd.DataFrame(np.zeros((1, len(c_visitantes))), columns=c_visitantes)
      elif agno1 > 0:
        agno1 -= 1

    local_mapping = {}
    for col_0, col_1 in zip(prt_nv_h.columns, c_locales):
      local_mapping[col_1] = [prt_nv_h[col_0].values[0]]
      
    visitor_mapping = {}
    for col_0, col_1 in zip(prt_nv_a.columns, c_visitantes):
      visitor_mapping[col_1] = [prt_nv_a[col_0].values[0]]

    if ind is None:
      data_metricas = {
          "home": [mnd.home],
          **local_mapping,
          "score_0": [mnd.score_0],
          "away": [mnd.away],
          **visitor_mapping,
          "score_1": [mnd.score_1],
      }
    elif ind == 0:
      data_metricas = {
          "home": [mnd.home],
          **local_mapping,
          "score_0": [mnd.score_0] if not (mnd.agno in (1934, 1938)) else [mnd.score],
          "away": [mnd.away],
          **visitor_mapping,
          "score_1": [mnd.score_1] if not (mnd.agno in (1934, 1938)) else [mnd.score],
      }
    agf_df = pd.DataFrame(data=data_metricas)
    df5 = pd.concat([df5, agf_df], axis=0)

  return df5.iloc[::-1].reset_index(drop=True)

class normali():
  def __init__(self, ind_partidos, ind_fixtures, df_prediccion_usuario, clsif, ind_grupos_mundiales=None, agno=None, obj0="-", evitar="-", embedding_dim=3):
    """
    ¿QUE HACE?
    Genera una serie de df y capas de embedding necesarios para dar funcionamiento a la busqueda de nombres de grupos para equipos, el orden de juego en cada mundial y correr la lógica de emparejamientos para los mundiales entre 1986 y 1994

    ¿COMO LO HACE?
    0-) Se coloca en una variable (df0) el resultado de la funcion "formar_dataset_real" ingresandole el argumento (ind_partidos), que será un df contenedor de todos los partidos de cierto mundial 

    1-) En el atributo de instancia (self.df2) se almacena una copia del df (df0) con el índice reseteado, conteniendo todos los partidos recuperados de la tabla indicada por (ind_partidos) 

    2-) Luego deja en un atributo de instancia (self.df2) una copia del o los mundiales recuperados, por "formar_dataset_real" 

    3-) En otro atributo de instancia (self.df3) se almacena el fixture del mundial de interés, usando la funcion "formar_dataset_real" con argumentos "fixtures" y el parametro (ind_fixtures) que contiene el año del mundial de interés en formato string, para la búsqueda del fixture correspondiente en la DB

    4-) Se crea un df contenedor (self.paises) de los nombres de todos los paises que jugaron en la fase de clasificacion previa al mundial de interés y los paises que juegan el mundial en si mismo 

    5-) PEND (buscar si se puede optimizar con tokenize)

    6-) Luego se deja en un atributo de instancia (self.obj0) el contenido del parametro (obj0) para indicar cuando se trabaja con o sin prediccion tras consulta de un usuario 

    7-) Se almacena en un atributo de instancia (self.df5) el df contendor de la consulta del usuario (df_prediccion_usuario) 

    8-) Se crea una capa embedding de keras indicando el tamaño de la entrada y salida de la misma para futura conversion de datos categoricos referentes a los nombres de los paises que juegan el mundial de interes

    9-) Si el parámetro (ind_grupos_mundiales) es diferente de "None" y el parámetro (agno) es diferente de "None"

      9.0-) Se define un atributo de instancia diccionario (self.dic_emp_0) con pares clave-valor con clave como el índice de la fila de un array y valor como grupos con los posibles mejores 3ros lugares tras la fase de grupos. Aplíquese la lógica: si los mejores terceros son de los grupos "clave":"valor", se tomará la "clave" para buscar en la fila de un array que sea igual a "clave" 

      9.1-) Se define un atributo de instancia array de grupos (self.emp_terceros) con 3ros lugares que jugarán contra los 1ros lugares de los grupos A,B,C,D para los partidos variables, según la "clave" a la que correspondan los mejores 4 3ros lugares de la fase de grupos

      9.2-) Se define un atributo de instancia lista de listas (self.partidos_10) que posee el las 4 posiciones fijas por grupo, de los equipos que jugarán en el 1er round de la fase final

      9.3-) Se define un atributo de instancia lista vacia (self.partidos_11) que contendrá las 4 posiciones variables por grupo, de los equipos que jugarán en el 1er round de la fase final

      9.4-) Se define un atributo de instancia diccionario (self.dic_emp_1) con pares clave-valor de clave los 3 años "1986", "1990", "1994" que estan como clave pero de tipo "int" y valor una tupla con dos tuplas internas, cada una con el momento de juego para cada partido fijo en la 1ra y variable en la 2da 
      
      9.5-) Se define un atributo de instancia tupla (self.ref_0) con 2 tuplas internas que almacena la combinación de valores obtenidos del atributo de instancia diccionario (self.dic_emp_1) en los que iran, según el año indicado como clave de busqueda con el argumento (agno), los partidos home y away fijos y variables, indicando el valor de cada elemento interno el orden de juego de cierto partido, fijo en caso de (self.partidos_10) y variable en caso de (self.partidos_11), en el 1er round de la fase final. E indicando el índice de dicho elemento, la ubicación dentro de las listas de almacenamiento de los partidos fijos y variables mencionadas, que tiene el partido a disputar

    - Args:

      - ind_partidos: valor de tipo str que indica a la funcion "formar_dataset_real" que tabla de la DB debe consultar
      
      - ind_fixtures: valor de tipo str que indica a la funcion "formar_dataset_real" que año debe buscar en la tabla fixture de la DB, proveniente de un slicing a los ultimos 4 elementos del argumento (pregunta_0) de la funcion "consulta_general"
      
      - df_prediccion_usuario: df consulta del usuario, básicamente un df de la fase con datos editados por usuario para hacer inferencia
      
      - clsif: df contenedor de partidos de clasificacion previa al mundial de interés a predecir
      
      - ind_grupos_mundiales: indica si debe o no entrarse en la lógica de emparejamiento por mejores 3ros lugares adoptada por la FIFA entre 1986 y 1994, siendo "None" en el caso de no requerirla y diferente de None (ej. "grupos") en el caso de que sí se requiera dicha lógica 
      
      - agno: valor int que indica el año en que se busca la ubicacion de cada partido fijo o variable, para la lógica de emparejamiento por mejores 3ros lugares de los mundiales 1986-1994, dentro del atributo de instancia diccionario (self.dic_emp_1), proveniente de un slicing a los ultimos 4 elementos del argumento (pregunta_0) de la funcion "consulta_general"
      
      - obj0: puede ser un str de la forma "-", o un slicing a los ultimos 4 elementos del argumento (pregunta_0) de la funcion "consulta_general", indicando en el 1er caso que el 1er valor de retorno de la funcion que usa este argumento desde el atributo de instancia (self.obj0), será igual a todos los partidos del mundial actual, e indicando para el 2do caso, que este 1er valor de retorno se divide entre los partidos de la fase final y los partidos del df de consulta del usuario
      
      - embedding_dim: valor indicador del tamaño de los vectores embedding para cada palabra
    """
    df0 = formar_dataset_real(ind_partidos)
    self.df2 = df0.copy().reset_index(drop=True)
    self.df3 = formar_dataset_real("fixtures", ind_fixtures)
    
    self.paises = pd.concat([self.df2.home, self.df2.away, self.df3.home, self.df3.away, clsif.home, clsif.away], axis=0)

    self.pais_id = {pais: id for id, pais in enumerate(self.paises.unique())}
    self.id_pais = {id: pais for pais, id in self.pais_id.items()}

    self.obj0 = obj0
    self.df5 = df_prediccion_usuario
    self.embedding_layer = tf.keras.layers.Embedding(input_dim=len(self.pais_id), output_dim=embedding_dim)

    if ind_grupos_mundiales and agno is not None:
      self.dic_emp_0 = {0: ['Grupo A', 'Grupo B', 'Grupo C', 'Grupo D'],
                        1: ['Grupo A', 'Grupo B', 'Grupo C', 'Grupo E'],
                        2: ['Grupo A', 'Grupo B', 'Grupo C', 'Grupo F'],
                        3: ['Grupo A', 'Grupo B', 'Grupo D', 'Grupo E'],
                        4: ['Grupo A', 'Grupo B', 'Grupo D', 'Grupo F'],
                        5: ['Grupo A', 'Grupo B', 'Grupo E', 'Grupo F'],
                        6: ['Grupo A', 'Grupo C', 'Grupo D', 'Grupo E'],
                        7: ['Grupo A', 'Grupo C', 'Grupo D', 'Grupo F'],
                        8: ['Grupo A', 'Grupo C', 'Grupo E', 'Grupo F'],
                        9: ['Grupo A', 'Grupo D', 'Grupo E', 'Grupo F'],
                        10: ['Grupo B', 'Grupo C', 'Grupo D', 'Grupo E'],
                        11: ['Grupo B', 'Grupo C', 'Grupo D', 'Grupo F'],
                        12: ['Grupo B', 'Grupo C', 'Grupo E', 'Grupo F'],
                        13: ['Grupo B', 'Grupo D', 'Grupo E', 'Grupo F'],
                        14: ['Grupo C', 'Grupo D', 'Grupo E', 'Grupo F'],
                        }

      self.emp_terceros = np.array([["Grupo C", "Grupo D", "Grupo A", "Grupo B"],
                                    ["Grupo C", "Grupo A", "Grupo B", "Grupo E"],
                                    ["Grupo C", "Grupo A", "Grupo B", "Grupo F"],
                                    ["Grupo D", "Grupo A", "Grupo B", "Grupo E"],
                                    ["Grupo D", "Grupo A", "Grupo B", "Grupo F"],
                                    ["Grupo E", "Grupo A", "Grupo B", "Grupo F"],
                                    ["Grupo C", "Grupo D", "Grupo A", "Grupo E"],
                                    ["Grupo C", "Grupo D", "Grupo A", "Grupo F"],
                                    ["Grupo C", "Grupo A", "Grupo F", "Grupo E"],
                                    ["Grupo D", "Grupo A", "Grupo F", "Grupo E"],
                                    ["Grupo C", "Grupo D", "Grupo B", "Grupo E"],
                                    ["Grupo C", "Grupo D", "Grupo B", "Grupo F"],
                                    ["Grupo E", "Grupo C", "Grupo B", "Grupo F"],
                                    ["Grupo E", "Grupo D", "Grupo B", "Grupo F"],
                                    ["Grupo C", "Grupo D", "Grupo F", "Grupo E"]])

      self.partidos_10 = [["Winner Grupo F", "Runners-up Grupo E"],
                          ["Runners-up Grupo A", "Runners-up Grupo C"],
                          ["Runners-up Grupo F", "Runners-up Grupo B"],
                          ["Winner Grupo E", "Runners-up Grupo D"]] 
      self.partidos_11 = [] 

      self.dic_emp_1 = {1986: ((1, 2, 5, 6), (0, 3, 4, 7)),
                        1990: ((7, 4, 2, 1), (0, 3, 5, 6)),
                        1994: ((3, 0, 7, 5), (1, 2, 4, 6))}
      self.ref_0 = self.dic_emp_1[agno]

  def cuatro_mejores_terceros(self, dic_g):
    """
    ¿QUE HACE?
    Genera y retorna una lista con valores string de los nombres de los grupos en que quedaron los equipos con 4 mejores puntajes, tras la fase de grupos. Esto se hace mediante la seleccion de todos los equipos 3ros de cada grupo segun su puntaje, el aislamiento de los puntajes de estos en una variable y la seleccion sucesiva durante 4 cicclos de los 4 mas altos puntajes de entre estos, agregando el grupo en el que se registraron a una variable (cuatro_mayores) que acabará retornandose

    ¿COMO LO HACE?
    0-) Se crean unas variables lista (ter_cd_grp, cuatro_mayores) vacias

    1-) Se iteran todos los df de tabla de posicion de cada grupo que ha jugado la fase de grupos y se almacena en (ter_cd_grp), como lista, los valores de nombre, puntaje y grupo del equipo que quedara como 3er lugar del df iterado

    2-) Se itera un bucle de 4 ciclos sin informacion y, se forma un df temporal (seleccion) con los 3ros lugares guardados en (ter_cd_grp) indicandose las columnas ("pais","Pts","Grupo") - se guarda en la variable (puntos_terceros) una lista con los valores de la columnas "Pts" de (seleccion) - se guarda en (mayor_vu) el Pts maximo de los registrados - se guarda en (seleccion) la fila del df (seleccion) previo, donde Pts es igual a (mayor_vu) - se añade a (cuatro_mayores) el grupo del mejor tercero registrado en (seleccion) - se quita de (ter_cd_grp) la lista con los valores de mejor tercero que se habria guardado en (seleccion)
    
    3-) Se retorna (cuatro_mayores) con los mejores 4 3ros que se hayan registrado en el bucle

    - Args:

      - dic_g: diccionario contenedor de pares clave-valor con clave como el nombre de alguno de los n grupos de equipos que disputan el mundial (Grupo_n) y valor como el df con la informacion de rendimiento particular de cada equipo, segun como apareceria en la tabla de posiciones

    - Returns:

      - cuatro_mayores: lista contenedora de los nombres de los 4 grupos en que se registraron los mejores 3eros lugares por puntaje tras los partidos en la fase de grupos
    """
    ter_cd_grp = []
    cuatro_mayores = []

    for des0 in dic_g.values():
      ter_cd_grp.append(des0.iloc[2, :][["pais", "Pts", "Grupo"]].tolist())

    for _ in range(4):
      seleccion = pd.DataFrame(data=np.array(ter_cd_grp), columns=["pais", "Pts", "Grupo"])
      puntos_terceros = seleccion.Pts.tolist()
      mayor_vu = max(set(puntos_terceros))
      seleccion = seleccion.loc[seleccion.Pts == mayor_vu]
      cuatro_mayores.append(seleccion["Grupo"].values[0])
      ter_cd_grp.remove(seleccion.values[0].tolist())

    return cuatro_mayores

  def emparejamiento_partidos_dependientes(self, cuatro_mayores):
    """
    ¿QUE HACE?
    Agrega a un atributo de almacenamiento (partidos_11), cuales serán los emparejamientos a darse para partidos del 1er round de la fase final 

    ¿COMO LO HACE?
    0-) Se iteran los pares clave, valor del atributo diccionario de emparejamientos (dic_emp_0)

    1-) Si hay coincidencia total, medida en un tamaño de 4 elementos coincidentes, entre el valor iterado del diccionario de emparejamientos (_[1]) y el parametro de mejores 3ros (cuatro_mayores)

      1.0-) Se guarda en una variable (emp_1) la fila correspondiente a la clave en que se hayo la coincidencia de 4 entre el parametro (cuatro_mayores) y el atributo (dic_emp_0), dentro del array atributo (emp_terceros)
    
    2-) Se iteran en un for con zip los pares de grupos ganadores fijos ["Grupo A", "Grupo B", "Grupo C", "Grupo D"] y los grupos de 3ros lugares variables presentes en (emp_1) - Se añade al atributo (partidos_11) una lista contenedora de los emparejamientos entre 1ros lugares de los grupos del A al D y los 3ros lugares correspondientes de los demas grupos

    - Args:

      - cuatro_mayores: lista contenedora de los nombres de los grupos en que se registraron los 4 mejores 3ros

    - Returns:

      No retorna nada, solo actualiza el parametro (partidos_11) añadiendo 4 listas con los pares ganador de cierto grupo, 3er lugar de cierto grupo, para 4 de los 8 partidos a disputarse en el 1er round de la fase final "usualmente 8vos de final aunque puede variar"
    """
    for _ in self.dic_emp_0.items():
      if len(set(_[1]).intersection(cuatro_mayores)) == 4:
        emp_1 = self.emp_terceros[_[0]]

    for des0, des1 in zip(["Grupo A", "Grupo B", "Grupo C", "Grupo D"], emp_1):
      self.partidos_11.append(["Winner " + des0, "Third-p " + des1])

  def octavos_1986_a_1994(self, ref_0):
    """
    ¿QUE HACE?
    Genera un df desde un array contenedor de todos los pares home away que jugarán los 1ros 8 partidos de la fase final, a partir del contenido de unas tuplas internas en (ref_0) que apuntan a posiciones en los parametros (partidos_11, partidos_10), y el posicionamiento dentro de (lista_8vos)

    ¿COMO LO HACE?
    0-) Se crea una lista (lista_8vos) con 8 valores "None" default, para posicionar listas pares de equipos home y away que jugarán los primeros 8 partidos de la fase final
    
    1-) Se itera un par de "enumerate" de cada tupla del parametro (ref_0), en unas variables (des0) y (des1)

    2-) En la lista (lista_8vos) se van almacenando partidos indicados segun el orden marcado para los atributos (partidos_11) y (partidos_10), por el orden de las variables de iteracion (des0) y (des1) de las tuplas de referencia del parametro (ref_0); el contenido de esas variables indica la posicion del partido en "lista_8vos"

    3-) Se retorna un df con data igual a un array de (lista_8vos) para los equipos home y away
    
    - Args:

      - ref_0: tupla con 2 tuplas internas con 4 valore cada una, indicando su posicion la posicion de de un par home y away dentro de las listas de partidos constantes y variables respectivamente, mientras que el contenido de cada tupla interna indica la posicion que adoptará ese partido especifico dentro de la variable de (lista_8vos) de 1er round de la fase final

    - Return:

      - Retorna un df con el contenido y orden de juego de los partidos del 1er round de la fase de grupos
    """
    lista_8vos = [None, None, None, None, None, None, None, None]

    for des0, des1 in zip(enumerate(ref_0[0]), enumerate(ref_0[1])):
      lista_8vos[des1[1]] = self.partidos_11[des1[0]]
      lista_8vos[des0[1]] = self.partidos_10[des0[0]]

    return pd.DataFrame(data=np.array(lista_8vos), columns=["home", "away"])

  def ajuste_score_partidos(self, x, dictionary, ind=0):
    """
    ¿QUE HACE?
    Sustituye caracteres basura resultantes del scraping, por los que convienen al preprocesamiento, entrenamiento e inferencia con los datos

    ¿COMO LO HACE?
    0-) Se crea un parametro (translate_table) que usa el metodo "str.maketrans" para dar soporte al metodo ".translate" con una tabla de referncia para uno a uno caracteres. Se crea un parametro (translated_text) para traducir caracteres de una cadena, segun (translate_table) mediante ".translate". Desde el paramtro (new_string) se obtiene una particion de la cadena (translated_text) tomando "," como separador

    1-) Si el parametro (ind) es igual a 0 o igual a 1

      1.0) Se define una variable (goles) que representa el numero de goles para el equipo home o away, segun el valor que tenga (ind), para buscar en (new_string)

      1.1) Retorna la variable (goles)
      
    Args:

        - x: string a limpiar

        - dictionary: diccionario con clave (caracter a reemplazar), valor (caracter de reemplazo)

        - ind: indica si retornaremos el resultado formateado segun el diccionario para los goles
              del equipo 0 o equipo 1 (score_0, score_1)

    Returns:

        - goles: el numero de goles para el equipo de interes como un integer
    """
    translate_table = str.maketrans(dictionary)
    translated_text = x.translate(translate_table)
    new_string = translated_text.split(",")
    
    if (ind == 0) or (ind == 1):
      goles = int(new_string[ind])
      return goles
    
  def hacer_embedding_a_equipos(self, entrada):
    """
    ¿QUE HACE?
    convierte a tensor los valores categoricos referentes a los nombres de los equipos que juegan cieta fase del mundial. Se concatena la version embedding como array tipo float de los nombres de los equipo de la fase trabajada, junto con los valores de rendimiento recuperados de (entrada) y se entrega ese nuevo array concatenado como resultado 

    ¿COMO LO HACE?
    0-) Se convierte a tensor los valores de (entrada) que corresponden a los nombres de los equipos que jugarian cierto round de cierta fase ya sea como local o visitante, pasandolos todos a int64 y guardandolo en (indices_tensor)

    1-) Referenciando a (self.embedding_layer) se accede al objeto de la funcion Embedding de tensorflow, se le pasa el (indices_tensor) como parametro y se guarda este formato embedding como array numpy en (embeddings_array)

    2-) En la variable (result) se concatena como columna el array de embeddings (embeddings_array) con el array de todos los valores de rendimiento (entrada[:,1:]), correspondientes a todas las columnas de (entrada) a partir de la posicion 1, omitiendo unicamente la columna 0 que contiene el id del equipo. Se retorna (result)

    - Args:

      - entrada: array numpy contendor del orden de juego por equipo local o visitante segun se pase y los valores de rendimiento de cada uno

    - Returns:

      - result: array concatenado del codigo embedding para cada equipo y sus valores de rendimiento, entresacados de (entrada)
    """
    indices_tensor = tf.convert_to_tensor(entrada[:, 0], dtype=tf.int64)

    embeddings = self.embedding_layer(indices_tensor)
    embeddings_array = embeddings.numpy()
    result = np.concatenate((embeddings_array, entrada[:, 1:].astype(float)), axis=1)
    return result

  def c_s(self, data):
    '''
    ¿QUE HACE?
    Recive un array en el parametro (data) contenedor de los partidos del round de la fase trabajada con los equipos codificados y ejecuta la eliminacion de la columna de "score_n", luego va guardando en unas listas vacias (X, y) los lotes de informacion segun el tamaño de paso establecido en ---n_steps0--- y su respuesta esperrada para el "score_n". Al final se devuelve una tupla con features y valores esperados referentes a los goles que estos equipos, con estos valores de rendimiento, habrian de anotar
    
    ¿COMO LO HACE?
    0-) Se crean unas listas vacias (X, y). Se elimina de (entrada) la columna correspondiente al score_n segun si se trabaja con equipo local o visitante
    
    1-) Se almacena en una variable (n_steps0) el tamaño de paso con que se definirán los lotes de informacion con que se alimentará el modelo para la inferencia

    2-) Se itera un rango desde la variable de iteracion (i) correspondiente al numero de filas que tendria el array pasado en (data), menos el tamaño de paso. Se guarda en la lista (X) el lote de filas con todas las columnas de informacion de equipo y rendimiento, desde que marca (i) hasta (i + n_steps0) y dejando para (y) justo el valor que corresponderia a las columnas de la fila (i + n_steps0). Finalmente se retorna una tupla con ambos arrays (X, y)
    
    - Args:

      - data: array contendor de nombres de los equipos codificados como embedding y sus valores de rendimiento 

    - returns:

      - np.array(X), np.array(y): tupla contenedora de identidad de cada equipo (nombre codificado, valores de rendimiento) en (X) y su correspondiente respuesta esperada en goles almacenada en "score_n" dentro de la lista (y)
    '''
    X, y = [], []
    data0 = np.delete(data, C_S_DELETE_INDEX, axis=1)

    n_steps0 = N_STEPS

    for i in range(len(data) - n_steps0):
      X.append(data0[i:i + n_steps0, :])
      y.append(data[i + n_steps0 - 1, C_S_Y_INDEX])

    return np.array(X), np.array(y)

  def grupos_anio_interes(self, year):
    """
    ¿QUE HACE?
    Toma los valores de una consulta a DB con ORM usando de parámetro la tabla y el valor de la columna, para pasar el nombre categorico a indice, en los partidos que habria judado un equipo, formar un diccionario con claves (nombre del grupo) valor (valores de rendimiento actualizados tras las eliminatorias para cada equipo del grupo). Finalmente, ese diccionario se retorna a la salida de la funcion

    ¿COMO LO HACE?
    0-) Se emplea la funcion "formar_dataset_real" para recuperar con el ORM de django, las filas de la base de datos postgresql, de la tabla grupos, donde la columna agno sea igual a (year). Se guarda toda esta informacion en una variable llamada (df0)

    1-) Toma del df devuelto en (df0) y transforma el contenido de la columan "pais" a indices para facilitar su compatibilidad con las entradas del modelo de deep learning

    2-) Se usa un ".loc" para afectar la direccion en memoria de todas las filas y columnas, menos la primera y la ultima, de (df0) igualando a 0 el conteniddo dentro de este rango
    
    3-) genera el diccionario (dic_0) con la iteracion de los valores unicos de la columna del (df0) correspondiente a los grupos para cada pais y deja como clave (nombre del grupo) y valor (valores de rendimiento actualizados tras las eliminatorias para cada equipo del grupo)
    
    - Args:

      - year: año del mundial del que interesa recuperar los grupos

    - Returns:

      - dic_0: diccionario contenedor del nombre de cada grupo como valor los valores de rendimiento para cada equipo del grupo, acumulados durante la fase eliminatoria previa al mundial
    """
    df0 = formar_dataset_real("grupos", year)
    df0["pais"] = df0["pais"].map(self.pais_id)
    metrics_to_zero = [col for col in df0.columns if col not in ("pais", "Grupo", "agno")]
    df0.loc[:, metrics_to_zero] = 0
    dic_0 = {des0: df0[df0["Grupo"] == des0] for des0 in set(df0["Grupo"].tolist())}
    return dic_0

  def create_features(self, dic_ag):
    """
    ¿QUE HACE?
    Arroja las secuencias necesarias tanto de la data historia del mundial de interes, como del fixture del mismo, apuntando a adaptarse al formato del torneo que tuvo ese mundial en particular
    
    Arrojará especificamente el historico de partidos de la fase final del mundial de interés (extraídos del df historico completo o acotados solo a la fase final segun el valor del atributo obj0), la secuencia definida por el usuario para hacer inferencia, la fase de grupos para el mundial de interés, la estructura de fixture de fase final, todo en ese orden. La idea de arrojar estos datos, es permitir la inferencia en el orden marcado por la estructura que tendria el mundial de interés para el usuario, presente desde el propio fixture de ese mundial

    Puede tambien arrojar solo el historico de partidos completo del mundial, el historico de partidos de la fase de grupos, el fixture de la fase final

    ¿COMO LO HACE?
    0-) Se iteran todos los grupos del mundial presentes en (dic_ag) y se actualiza en la columna "pais" al indice de este segun el atributo (pais_id)

    1-) Se crea una variable (df01) y se le asigna una copia del contenido del mundial de interés en el atributo (df2). Se crea una variable (year0) que almacena como una serie, la columna del año o "year" de (df2). Con (pais_id) se actualizan los nombres de los equipos que juegan como home y away en el df (df01)

    2-) (data_limpia_0, data_limpia_1) se almacenan series correspondientes a la division entre goles del equipo home y goles del equipo away para todos los partidos registrados en (df2), mediante el uso del metodo (ajuste_score_partidos)

    3-) Se crea un atributo df (df1) para guardar el nombre id de los equipos home y away, los puntajes guardados en (data_limpia_0, data_limpia_1) aqui etiquetados como "score_0, score_1", el año de juego del mundial de itneres y la especificacion de indices respecto al tamaño del mundial presente en (df01)

    4-) Se actualiza en (df1) todas las metricas de rendimiento desarrolladas por cada equipo home y away, during the fase de clasificaciones, usando la funcion "agregar_features". Se crea una variable (df02) con una copia del fixture del mundial actual

    5-) Mediante una referencia index con negacion booleana (~) se obtiene toda la seccion del (df02) que contiene la fase de grupos de el mundial de interes, luego se obtiene el tamaño de esta fase y se almacena en (longitud_fase_grupos). Mediante este valor numerico se obtiene un slicing con todos los partidos de fase de grupos y se actualiza con estos datos, la variable (df02)

    6-) Se crean 2 condicionales

        6.0) Si el año del fixture del mundial actual es 1934 o 1938. Se actualizan los nombres de los equipos home y away a su indice segun el atributo (pais_id) en la variable (df02)

        6.1) Si el año del fixture del mundial actual no es 1934 o 1938. Si el año del fixture del mundial actual es 1934 o 1938. Se actualizan los nombres de los equipos home y away a su indice segun el atributo (pais_id) en la variable (df02). En unas variables (data_limpia_2, data_limpia_3) se almacenan series correspondientes a la division entre goles del equipo home y goles del equipo away para todos los partidos registrados en la fase de grupos de (df3), mediante el uso del metodo (ajuste_score_partidos)

          6.1.0) Se actualiza (df02) a un df con los nombres de los equipos home y away que juegan la fase de grupos del mundial de interes, los puntajes de cada partido "score_0" y "score_1", el año de ese mundial y los indices del df usando el tamaño del df (df02)

    7-) Se actualizan las metricas de rendimiento del df (df02) dentro de esa misma variable, usando la funcion "agregar_features". Se crean unas variables (df2, df3) que seran una copia de la fase de grupos del fixture (df02) y un slice de la fase final del fixture (df02) 
    
    8-) Se crean 2 condicionales 

      8.0) Si el atributo (obj0) indica que se tomarán datos del usuario. Se actualiza el df (df1) a los partidos de fase final. En el atributo (df5) con el df enviado por el usuario se actualizan los nombres de los equipos home y away a su forma id. Se actualiza (df1) a una tupla con la fase final no fixture en (df1) y el df del usuario en el atributo (df5). Se retorna: (df1) con fase final y df del usuario, (df2) con la fase de grupos del mundial y (df3) con la fase final del fixture

      8.1) Se retornan: (df1) se deja como contenedor de los partidos del mundial, df2 con el contenido de la fase de grupos en el fixture y df3 con el contenido de la fase final en el fixture

    Args:

        - dic_ag: diccionario con las metricas de rendimiento de cada equipo segun lo jugado en la fase de clasificaciones

    Returns: 

        - CASO 1: (df1, df2, df3). (fase final del mundial jugado, df consulta del usuario), (fase de grupos del fixture), (fase final del fixture) 

        - CASO 2: (df1, df2, df3). (todos los partidos con formato correcto del mundial jugado), (fase de grupos del fixture), (fase final del fixture)
    """
    for des0 in dic_ag:
      dic_ag[des0].pais = dic_ag[des0].pais.map(self.pais_id)

    df01 = self.df2.copy()
    year0 = self.df2.year
    df01.home = df01.home.map(self.pais_id)
    df01.away = df01.away.map(self.pais_id)
    data_limpia_0 = pd.Series(df01["score"]).map(lambda x: self.ajuste_score_partidos(x, {"(": r"", ")": r"", ":": r",", " ": r","})).to_list()
    data_limpia_1 = pd.Series(df01["score"]).map(lambda x: self.ajuste_score_partidos(x, {"(": r"", ")": r"", ":": r",", " ": r","}, ind=1)).to_list()

    df1 = pd.DataFrame({"home": df01.home.to_list(), "score_0": data_limpia_0, "score_1": data_limpia_1, "away": df01.away.to_list(), "agno": year0.to_list()}, index=[list(range(len(df01)))])
    
    df1 = agregar_features(df1, dic_ag)

    df02 = self.df3.copy()
    longitud_fase_grupos = df02[~df02["away"].map(lambda x: any(term in x for term in ["Match", "Grupo"]))].shape[0]
    df02 = self.df3.iloc[:longitud_fase_grupos, :]
    
    if df02.year.unique().tolist()[0] in (1934, 1938):
      df02.home = df02.home.map(self.pais_id)
      df02.away = df02.away.map(self.pais_id)
      
    elif df02.year.unique().tolist()[0] not in (1934, 1938):
      df02.home = df02.home.map(self.pais_id)
      df02.away = df02.away.map(self.pais_id)
      data_limpia_2 = pd.Series(df02["score"]).map(lambda x: self.ajuste_score_partidos(x, {"(": r"", ")": r"", ":": r",", " ": r","}))
      data_limpia_3 = pd.Series(df02["score"]).map(lambda x: self.ajuste_score_partidos(x, {"(": r"", ")": r"", ":": r",", " ": r","}, ind=1))

      df02 = pd.DataFrame({"home": df02.home.to_list(), "score_0": data_limpia_2, "score_1": data_limpia_3, "away": df02.away.to_list(), "agno": year0.to_list()[:longitud_fase_grupos]}, index=list(range(len(df02))))

    df02 = agregar_features(df02, dic_ag, ind=0)
    df2, df3 = df02.copy(), self.df3.iloc[longitud_fase_grupos:, :]
    
    if self.obj0 != "-":
      df1 = df1.iloc[longitud_fase_grupos:, :]

      self.df5.home = self.df5.home.map(self.pais_id)
      self.df5.away = self.df5.away.map(self.pais_id)

      df1 = (df1, self.df5)
      return df1, df2, df3

    else:
      return df1, df2, df3

def emparejar_equipos(ObjNrml):
  """
  ¿QUE HACE?
  Se genera un df contenedor de los 1ros n partidos de la fase final, mediante el uso de los metodos de emparejamiento de la clase "normali". Especificamente, se accede a todos los df de los grupos de un mundial, se saca a cada uno su 3er lugar y se determina cual tuvo los 4 mejores 3ros, se generan los emparejamientos variables entre 1ros y mejores 3ros lugares y finalmente se crea un df con todos los emparejamientos del 1er round de la fase final, tanto los variables (1ros vs mejores 3ros determinados por reglamento FIFA) como los estaticos (cruces fijos predefinidos por reglamento FIFA para el formato 1986-1994)
  
  ¿COMO LO HACE?
  0-) Genera un diccionario con pares clave igual a los grupos que hay en el mundial y valor igual a los df de rendimiento vacios de dichos grupos y lo guarda en una variable (dic_g), todo mediante un llamamiento al método "grupos_anio_interes" y el uso del atributo "agno" de la clase "normali" como parámetro del atributo

  1-) Se emplea el atributo "cuatro_mejores_terceros" para dejar en una variable (cuatro_mayores) una lista con los grupos en los que quedaron los 4 mejores equipos en 3er lugar segun su puntaje

  2-) Se emplea el método "emparejamiento_partidos_dependientes" de la clase "normali" para dejar en el atributo lista (partidos_11) una serie de 4 listas contenedoras de pares de equipos expresados en la forma ["Winner Grupo n", "Third-p Grupo n"] representando los 4 partidos con emparejamiento variable, cuyo rival del 3er lugar depende de cuales sean los 4 mejores 3ros (recordando que habrán 4 partidos con emparejamiento fijo predeterminado por reglamento FIFA para el formato jugado desde 1986 hasta 1994)

  3-) Mediante el método "octavos_1986_a_1994" y el contenido del atributo (ref_0) que estará influido por el año del mundial estudiado, se genera un df contendor de todos los partidos del 1er round de la fase final, segun los pares de equipos almacenados en los atributos (partidos_10) y (partidos_11) cuyo contenido se reposicionará en el df partido a partido en el orden segun el contenido y de los elementos del (ref_0)

  Args:

    - ObjNrml: objeto de la clase "normali" para acceder al sistema de emparejamientos adoptado
               por la FIFA entre 1986 y 1994

  Returns:

    - df con el orden de juego de equipos home y away en los partidos del 1er round de la fase final del mundial estudiado
  """
  dic_g = ObjNrml.grupos_anio_interes(ObjNrml.agno)
  cuatro_mayores = ObjNrml.cuatro_mejores_terceros(dic_g)
  ObjNrml.emparejamiento_partidos_dependientes(cuatro_mayores)
  return ObjNrml.octavos_1986_a_1994(ObjNrml.ref_0)

def knock_out_1986_1994(df0, df1):
  """
  ¿QUE HACE?
  Toma la salida de "emparejar_equipos" (df0) y el contenido del df consulta del usuario en (df1) y toma como lista los valores de "home" y "away" del 1ro, mientras que toma como lista los valores "score" del 2do, alamacena los 2 primeros en variables (emp0) y (emp1), y el "score" en una variable (sc_emp). Luego retorna un df contenedor de estas variables ya en orden y con el indice apropiado para cada fila
  
  ¿COMO LO HACE?

  0-) (emp0) y (emp1) almacenaran respectivamente la columna del "home" y "away" del df de 1er round de fase final que, de hecho, solo posee esas 2 columnas, convertido en serie y luego en lista

  1-) En (sc_emp) se almacena la columna "score" del df (df1) para las 1ras 8 filas, convertido a serie y luego a lista

  2-) se retorna un df con columnas "home", "score" y "away" con contenido igual a las variables (emp0), (sc_emp) y (emp1) respectivamente y con index de 0 al tamaño de (sc_emp)-1

  Args:

    - df0: df con el orden de juego de los equipos en sus unicas columnas "home" y "away" para 1ros 8 partidos de la fase final

    - df1: df de consulta del usuario con columna de score posiblemente editada por este para los 1ros 8 partidos de la fase final

  Return:

    - df con columnas "home", "score" y "away" con contenido igual a las variables (emp0), (sc_emp) and (emp1) respectivamente y con index de 0 al tamaño de (sc_emp)-1
  """
  emp0 = df0[["home"]].squeeze().to_list()
  emp1 = df0[["away"]].squeeze().to_list()
  sc_emp = df1[["score"]].iloc[:8].squeeze().to_list()

  return pd.DataFrame({"home": emp0, "score": sc_emp, "away": emp1}, index=list(range(0, len(sc_emp))))

class func_prediccion_orden(normali):
  def __init__(self, ind_partidos, ind_fixtures, df_prediccion_usuario, clsif, ind_grupos_mundiales=None, agno=None, obj0="-", evitar="-", embedding_dim=3, ind0=[None, None], n_partidos=0):
    """
    ¿QUE HACE?
    Inicializa la clase (func_prediccion_orden), que hereda de (normali), llamando al constructor del padre y definiendo los atributos adicionales necesarios para gestionar el flujo de predicciones secuenciales entre fases del mundial: la lista ordenada de fases, el diccionario de resultados por partido, el diccionario de fuerzas de equipos, el indicador de fase actual y objetivo, y el contador de partidos procesados

    ¿COMO LO HACE?
    0-) Se llama al constructor de la clase padre (normali) pasandole todos los parametros recibidos para inicializar los atributos heredados de esa clase

    1-) Se define el atributo lista (self.lista_fases) con los nombres de las fases del mundial en el orden en que se juegan, para su uso como referencia de comparacion entre la fase actual y la fase objetivo de la inferencia

    2-) Se inicializa el atributo diccionario (self.dic_matches) en None, para su posterior llenado con los resultados (ganador y perdedor) de cada partido procesado

    3-) Se inicializa el atributo diccionario (self.dic_fuerza) vacio, para su posterior llenado con los promedios de goles de cada equipo como local y visitante

    4-) Se almacena en el atributo (self.ind0) el parametro (ind0), que es una lista de 2 elementos indicando respectivamente la fase actual de inferencia (ind0[0]) y la fase objetivo marcada por el usuario como inicio de prediccion (ind0[1])

    5-) Se almacena en el atributo (self.n_partidos) el parametro (n_partidos), que lleva la cuenta acumulada del numero total de partidos procesados

    Args:

      - ind_partidos: valor de tipo str que indica a la funcion "formar_dataset_real" que tabla de la DB debe consultar

      - ind_fixtures: valor de tipo str con el año del mundial de interés para indicar la búsqueda correcta del fixture en la DB

      - df_prediccion_usuario: df de consulta del usuario con los partidos y puntajes editados para hacer inferencias personalizadas

      - clsif: df contenedor de partidos de clasificacion previa al mundial de interés

      - ind_grupos_mundiales: indica si debe entrarse en la logica de emparejamiento por mejores 3ros adoptada entre 1986 y 1994, siendo None si no se requiere y un valor distinto de None (ej. "grupos") si se requiere

      - agno: valor int que indica el año del mundial con formato de emparejamiento por mejores terceros, para la búsqueda en (self.dic_emp_1)

      - obj0: cadena que indica si el primer valor de retorno de (create_features) incluirá todos los partidos del mundial o solo la fase final separada del df de consulta del usuario

      - evitar: parametro reservado para uso futuro

      - embedding_dim: valor indicador del tamaño de los vectores embedding para cada equipo

      - ind0: lista de 2 elementos indicando la fase actual de inferencia (ind0[0]) y la fase objetivo marcada por el usuario (ind0[1])

      - n_partidos: numero inicial de partidos ya procesados, generalmente 0 al inicio

    Return:

      - No retorna valor, solo inicializa los atributos de la instancia
    """
    normali.__init__(self, ind_partidos, ind_fixtures, df_prediccion_usuario, clsif, ind_grupos_mundiales=ind_grupos_mundiales, agno=agno, obj0=obj0, evitar=evitar, embedding_dim=embedding_dim)

    self.lista_fases = ["grupo", "doce", "knockout", "quarter", "semi", "third", "final", "Fase Final"]
    self.dic_matches = None
    self.dic_fuerza = {}
    self.ind0 = ind0
    self.n_partidos = n_partidos

  def intsc_prob_goles(self, X, model, alterno=0):
    """
    ¿QUE HACE?
    Realiza la inferencia hacia la entrada array (X) con el modelo (model). Obtiene la probabilidad de que un resultado del modelo sea factible mediante el "metodo de interseccion de Poisson". Usa las probabilidades  de Poisson arrojadas para sacar el puntaje por el metodo convencional dado para la fase de grupos (3 puntos partido ganado, 0 puntos partido perdido, 1 punto partido empatado para cada equipo). Tambien puede entregar los resultados default de (X) segun si (alterno) es 0 o 1 respectivamente

    ¿COMO LO HACE?
    0-) Se definen unas variables (X0, X1) contenedoras de la traduccion y formateo de la entrada array (X), a una entrada compatible con el modelo, donde la columna de id de cada pais es traducida mediante un metodo embedding "hacer_embedding_a_equipos" y toda la entrada cambia su forma para ser de 3 dimensiones con el metodo "c_s" segun el ----tamaño de paso---- que se quiera para una inferencia

    1-) hay 2 condicionales

      1.0) si (alterno) es igual a 0. El modelo ejecuta la prediccion sobre los valores de rendimiento para cada equipo y las almacena en unas variables (Y_0, Y_1). Se concatena como columna la salida de esa inferencia por el modelo en una variable (Y0)

      1.1) si (alterno) es igual a 1. Se guarda en unas variables (Y_0, Y_1) los resultados default que carga la entrada array (X) en las columnas correspondientes. Se actualizan esas variables para hacerlahs un vector 2-dimensional de n filas, 1 columna. Se concatenan ambos resultados como columna a la variable (Y0)

    2-) Se itera al mismo tiempo la entrada (X) y la inferencia o valores default (Y0). Se almacena en unas variables (home, away) los valores de desempeño de los equipos home y away. Se guarda en unas variables (y_home, y_away) los resultados default o arrojados por la inferencia del modelo.

    3-) Se tienen 2 condicionales

      3.0) Si los valores de (home, away) estan en el atributo diccionario de fuerzas dic_fuerza. Se multiplican los primedios de goles por equipo a enfrentarse para obtener el correspondiente lambda de la distribucion de Poisson y se almacena un una variable (prob_puntaje). Se almacena en unas variables (p_hm, p_aw) la probabilidad de Poisson segun (prob_puntaje) y la salida iterada de (Y0). Se multiplican ambas probabilidades y se dejan en una variable (p)

        3.0.0) Se tienen 2 condicionales

          3.0.0.0) Si (p) redondeada es mayor o igual a 0.5. Se tienen 2 condicionales

            3.0.0.0.0) Si (p_h) es mayor que (p_a). p_hm es igual a 3, p_aw es igual a 0

            3.0.0.0.1) Si (p_a) es mayor que (p_h). p_aw es igual a 3, p_hm es equal a 0

          3.0.0.1) Si (p) redondeada es menor a 0.5

            3.0.0.1.0) (p_hm) es igual a 1, (p_aw) es igual a 1

        3.0.1) Se entrega un iterable con el numero de goles predicho o default para home y away respectivamente, y el puntaje determinado con la distribucion de Poisson para home y away respectivamente

      3.1) Se retorna una tupla (0, 0)

    Args:

      - X: array con las entradas (home, away) referentes a cada partido de la fase del mundial estudiada

      - model: modelo entrenado listo para hacer inferencia

      - alterno: pametro predefinid a 0 para decidir si el metodo realiza inferencia con el modelo o arroja los valores default (esto se aplica cuando no se ha llegado a la fase objetivo marcada por el usuario para iniciar la inferencia)

    Return:
      - yield((y_home, points_home), (y_away, points_away)): un iterable con unas tuplas de 2 valores; el numero de goles default o predicho por el modelo para home y away respectivamente y la probabilidad de anotacion del home y away respectivamente

      - (0, 0): una tupla con 2 valores (0,0) para el caso de tener equipos que no hayan participado en mundiales y no haya datos sobre su fortaleza o debilidad.
    """
    cols_per_team = 1 + len(ALL_METRICS) + 1
    
    if alterno == 0:
      X0, X1 = self.c_s(self.hacer_embedding_a_equipos(X[:, :cols_per_team])), self.c_s(self.hacer_embedding_a_equipos(X[:, cols_per_team:]))
      Y_0, Y_1 = model.predict(X0[0]), model.predict(X1[0])
      Y0 = np.concatenate((Y_0, Y_1), axis=1)

    elif alterno == 1:
      idx_y0 = 6
      idx_y1 = cols_per_team + 6
      Y_0, Y_1 = X[:, idx_y0], X[:, idx_y1]
      Y_0, Y_1 = np.reshape(Y_0, (-1, 1)), np.reshape(Y_1, (-1, 1))
      Y0 = np.concatenate((Y_0, Y_1), axis=1)

    for resultados_Y0, entradas_X in zip(Y0, X):
      home = entradas_X[MATCH_HOME_IDX]
      away = entradas_X[MATCH_AWAY_IDX]

      y_home = resultados_Y0[0]
      y_away = resultados_Y0[1]

      p_hm, p_aw = 0, 0

      if home in self.dic_fuerza["home"].index and away in self.dic_fuerza["away"].index:
        prob_puntaje = self.dic_fuerza["home"].at[home, "goles"] * self.dic_fuerza["away"].at[away, "goles"]

        p_h = poisson.pmf(y_home, prob_puntaje)
        p_a = poisson.pmf(y_away, prob_puntaje)
        p = p_h * p_a

        if round(p) >= 0.5:
          if p_h > p_a:
            p_hm, p_aw = 3, 0
          elif p_a > p_h:
            p_aw, p_hm = 3, 0

        if round(p) < 0.5:
          p_hm, p_aw = 1, 1

        yield ((y_home, p_hm), (y_away, p_aw))

      else:
        yield (0, 0)

  def generar_matches(self, paso):
    """
    ¿QUE HACE?
    Deja en el atributo diccionario "self.dic_matches" otros diccionarios contenedores del ganador y perdedor por partido

    ¿COMO LO HACE?
    0-) Se define una variable (value_in) igual al numero de partidos sobre los que se ha hecho ya la inferencia, referenciando al atributo (n_partidos). Se define una variable (value_fin) igual al numero de partidos con los que se ha trabajado mas el numero de partidos se va a inferir con el modelo ahora "la nueva fase".

    1-) Se actualiza el atributo (dic_matches) con estructura (clave: partido (numero iterado del partido de la fase actual)), valor: (clave: ganador-perdedor, None-None)

    2-) Se actualiza el atributo (n_partidos) a (value_fin)

    - Args:
    
      - paso: numero que indica el tamaño de paso para dar nombre a los partidos del siguiente 
              encuentro

    - Return:

      - no retorna mas que un cambio de diccionario en "self.dic_matches" y actualiza el
        atributo "self.n_partidos" con el numero de indicador del primer partido del siguiente encuentro
    """
    value_in = self.n_partidos
    value_fin = value_in + paso
    self.dic_matches = {**{f"partido {col0}": dict(ganador=None, perdedor=None) for col0 in range(value_in, value_fin)}}
    self.n_partidos = value_fin

  def fase_de_grupos(self, dic_t, df_fixture_, model):
    """
    ¿QUE HACE?
    Ejecuta la inferencia con el modelo para los partidos de la fase de grupos, usandose una funcion específica para esta fase porque el modo de eliminacion y seleccion dado aqui es diferente al de la fase final que cumple con los parametros de un torneo de eliminacion simple, mientras que la fase de grupos es un torneo tipo round robin
    
    Para hacerlo, se vale de la seccion del fixture que incluye los partidos de esta fase y del diccionario que incluye a los df con la informacion del rendimiento para cada equipo de cada grupo

    ¿COMO LO HACE?
    1-) Se crea una variable array de recepcion de puntajes por partido (x)

    2-) Se iteran las clave-valor del diccionario (dic_t). Se obtienen los paises de la columna "pais" del df del grupo iterado y se dejan en una variable (paises). Se define un df (df_fix_group_n) que será seccion del (df_fixture_) con los partidos en que jugaran, como home o away, los equipos almacenados en la variable (paises). se definen y agregan las tasas de goles anotados y recibidos mas la diferencia entre ambos, para los equipos home y away. Se define una variable array copia de (df_fix_group_n) con el nombre (X)

    3-) Se tienen 2 condicionales

      3.0) Si la fase actual "la de grupos" está antes de la fase objetivo para una prediccion. Se almacena en una variable (puntos) el resultado de aplicar intsc_prob_goles a (X) con el parámetro "alterno" seteado como 1 "lo que indica que no se predice y solo se toma el historico de resultados para los partidos previos a la fase desde la que se quiere predecir" lo que arrojará tanto los goles por partido como el puntaje obtenido para cada equipo home y away

      3.1) Si la fase de grupos es la fase objetivo entonces. Se define una variable (puntos) como el resultado de intsc_prob_goles, con el parametro alterno inicializado como 0, "indicando que se ejecutará la inferencia con el modelo" lo que arrojará la prediccion de goles y el puntaje obtenido para cada equipo home y away

    4-) Se iteran en conjunto los puntajes para cada partido y los partidos sobre los que se infirió. Se toman los nombres de los equipos home y away. Para el home y el away del partido, se guarda como una fila en el array de recepcion (x) los goles predichos o arrojados por intsc_prob_goles. Se actualiza el puntaje con lo arrojado por intsc_prob_goles, dentro del df que referencia el (dic_t) para el grupo y los equipos home y away iterados

    5-) En el df que referencia el grupo iterado dentro de (dic_t) se reordenan las filas del mayor al menor puntaje. Se deja en ese df solo las columnas de pais y Pts. Se redondean los resultados numericos en el df a 0 puntos decimales

    6-) Los puntajes almacenados en (x) se almacenan en las columnas "score_0" y "score_1" del df_fixture_ de fase de grupos. Se actualizan las metricas de rendimiento mediante la funcion "calculo_metricas_1". Se retorna el (dic_t) actualizado con los nuevos puntajes

    Args:

      - dic_t: diccionario con clave "grupo del mundial" (Grupo X) valor "df con nombres y metricas de rendimiento de cada equipo
      
      - df_fixture_: df con el orden de juego y nombres de cada equipo para la fase de grupos
      
      - model: model para hacer la inferencia

    Returns:

      - dic_t: es el mismo diccionario (dic_t) que entra como parámetro pero con el puntaje de cada equipo actualizado

      - df_fixture_: es el mismo df (df_fixture_) que entra como parámetro pero con los parametros de rendimiento actualizados

    """
    x = np.array([])

    for group, dfs in dic_t.items():
      paises = dfs['pais'].values
      df_fix_group_n = df_fixture_[(df_fixture_['home'].isin(paises)) & (df_fixture_['away'].isin(paises))]

      df_fix_group_n = calculo_metricas_0(df_fix_group_n)

      X = df_fix_group_n.copy().to_numpy()
      
      if self.lista_fases.index(self.ind0[0]) < self.lista_fases.index(self.ind0[1]):
        puntos = self.intsc_prob_goles(X, model, alterno=1)
      else:
        puntos = self.intsc_prob_goles(X, model, alterno=0)

      for points_, row in zip(puntos, df_fix_group_n.itertuples()):
        home, away = row.home, row.away
        x = np.append(x, [points_[0][0], points_[1][0]])
        dic_t[group].loc[dic_t[group]['pais'] == home, 'Pts'] += points_[0][1]
        dic_t[group].loc[dic_t[group]['pais'] == away, 'Pts'] += points_[1][1]

      dic_t[group] = dic_t[group].sort_values('Pts', ascending=False).reset_index(drop=True)
      dic_t[group] = dic_t[group][['pais', 'Pts']]
      dic_t[group] = dic_t[group].round(0)

    df_fixture_["score_0"] = x[0::2]
    df_fixture_["score_1"] = x[1::2]

    calculo_metricas_1(df_fixture_)

    return dic_t

  def get_winner(self, df_fixture_updated, *args):
    """
    ¿QUE HACE?
    Ejecuta la inferencia para los partidos de la fase final, empleandose una funcion completa para abarcar la inferencia solo en esta fase porque el formato de la misma, para eliminacion y seleccion entre cada round, es diferente al que se tiene para la fase de grupos
    
    Para hacerlo recurre a la actualizacion del fixture formateado entregado por parametro y al modelo que ejecuta la inferencia, ademas del registro del ganador y perdedor para cada partido, en el atributo "dic_matches"

    ¿COMO LO HACE?
    0-) Se define un array (Y0) para recepcion de informacion y un string vacio (partido). Se guarda el numero de filas del df de parametro (df_fixture_updated). Se genera el diccionario correspondiente a las victorias y derrotas para los partidos a disputar segun el (df_fixture_updated). Se actualiza el (df_fixture_updated) para que contenga las tasas de goles anotados y recibidos para los home y away, todo dentro de una nueva variable (df_fixture_partidos). Se guarda en una variable (X) un array salido de la copia del (df_fixture_partidos)

    1-) Se tienen 2 condicionales
      
      1.0) cuando el (X) es de dimension 2
        
        1.0.0) Se tienen a su vez, 2 condicionales

          1.0.0.0) Si la fase actual esta antes de la fase objetivo para iniciar la inferencia. Se almacena en la variable (puntos) la impresion de resultados default del fixture para antes de la inferencia

          1.0.0.1) Si la fase actual es la objetivo o despues de esta. Se almacena en la variable (puntos) la inferencia del modelo para el array (X)
      
        1.0.1) Se itera el contenido de la variable (puntos) y las filas del df (df_fixture_updated) al mismo tiempo. Se almacena en las variables (home, away) el nombre de los equipos que jugaron como home y away segun el df (df_fixture_updated). Se almacena en las variables (point_h, point_a) los goles arrojados por el metodo de inferencia "intsc_prob_goles"

        1.0.2) Se tienen 2 condicionales

          1.0.2.0) Si el puntaje del home es mayor que el del away. Se deja la variable (ganador) como "home" y la variable (perdedor) como "away"

          1.0.2.1) Si el puntaje del home es menor que el del away. Se deja la variable (ganador) como "away" y la variable (perdedor) como "home"

        1.0.3) Se va almacenando en la variable (indice) el numero de partido iterado. Se almacena en la variable (partido) el nombre del partido con el formato (partido X). Se almacena en las claves "ganador" y "perdedor" de ese partido en (dic_matches) a las variables (ganador, perdedor). En el array (Y0) se almacena como fila, a los puntajes home y away

        1.0.4) para el df (df_fixture_updated) se actualizan las columans "score_0" y "score_1". Al df (df_fixture_updated) se le actualizan las metricas de rendimiento dado el resultado arrojado por el metodo de inferencia "intsc_prob_goles"

      1.1) cuando el (X) es de dimension 1. Se deja en la variable (índice) la resta entre el nuero de partidos total y el numero de partidos en la fase actual. Se deja en la variable (partido) el nombre del partido actual

      1.2) Se tienen 2 condicionales

        1.2.0) Si la fase actual esta antes de la fase objetivo para iniciar la inferencia. Se almacena en la variable (puntos) la impresion de resultados default del fixture para antes de la inferencia

        1.2.1) Si la fase actual es la objetivo o despues de esta. Se almacena en la variable (puntos) la inferencia del modelo para el array (X)

      1.3) Se itera el contenido de la variable (puntos). Se almacena en unas variables (home, away) el nombre de los equipos que jugaron como home y away en el partido estudiado. Se almacena en unas variables (point_h, point_a) el numero de goles que arroja el metodo de inferencia para los equipos home y away

      1.4) Se tienen 2 condicionales

        1.4.0) Si (point_h) es mayor que (point_a) se deja la variable (ganador) como home y la variable (perdedor) como away
        
        1.4.1) Si (point_h) es menor que (point_a) sw deja la variable (ganador) como away y la variable (perdedor) como home
        
      1.5) Se actualizan las claves "ganador", "perdedor" para el partido actual en (dic_matches). Se actualiza el array (Y0) con el numero de goles arrojado por "intsc_prob_goles" 

      1.6) para el df (df_fixture_updated) se actualizan las columans "score_0" y "score_1". Al df (df_fixture_updated) se le actualizan las metricas de rendimiento dado el resultado arrojado por el metodo de inferencia "intsc_prob_goles"
      
    2-) Se crean unas variables (homes, aways, Winners, Losers) alojadoras de unas listas vacias. Se iteran los partidos de la fase actual (df_fixture_updated). Se va almacenando en la variable (indice) el numero de partido iterado. Se almacena en la variable (partido) el nombre del partido con el formato (partido X). Se actualizan estas listas vacias con los nombres de los equipos home, away, ganador y perdedor del partido actual

    3-) Se crea un df (df_nombres) con los nombres de los equipos que desputan, ganan y pierden los partidos de la fase actuql. Se imprime dicho df (df_nombres)

    4-) Se retorna el df (df_fixture_updated)
    
    Args:

      - df_fixture_updated: df con los partidos de la fase actual
   
      - model: modelo ara hacer inferencia sobre los partidos de la fase actual en (df_fixture_updated)

    Return:

      - df_fixture_updated: df actualizado de la fase actual, con el resultado arrojado por el metodo de inferencia "intsc_prob_goles" y las metricas de rendimiento actualizadas con "calculo_metricas_1"
    """
    if len(args) == 1:
      model = args[0]
    elif len(args) == 2:
      model = args[1]
    else:
      raise TypeError("get_winner() expects the model parameter.")

    Y0 = np.array([])
    partido = ""
    tamagno = df_fixture_updated.shape[0]
    self.generar_matches(tamagno)

    df_fixture_partidos = calculo_metricas_0(df_fixture_updated)
    X = df_fixture_partidos.copy().to_numpy()

    if len(X.shape) == 2:
      if self.lista_fases.index(self.ind0[0]) < self.lista_fases.index(self.ind0[1]):
        puntos = self.intsc_prob_goles(X, model, alterno=1)
      else:
        puntos = self.intsc_prob_goles(X, model, alterno=0)

      for points_, row in zip(puntos, df_fixture_updated.itertuples()):
        home, away = row.home, row.away
        point_h, point_a = points_[0][0], points_[1][0]
        
        if point_h > point_a:
          ganador = home
          perdedor = away
        else:
          ganador = away
          perdedor = home
        
        indice = (self.n_partidos - tamagno) + row.Index
        partido = f"partido {indice}"
        self.dic_matches[partido]["ganador"] = ganador
        self.dic_matches[partido]["perdedor"] = perdedor

        Y0 = np.append(Y0, [point_h, point_a])

      df_fixture_updated["score_0"], df_fixture_updated["score_1"] = Y0[0::2], Y0[1::2]
      df_fixture_updated = calculo_metricas_1(df_fixture_updated)

    elif len(X.shape) == 1:
      indice = self.n_partidos - tamagno
      partido = f"partido {indice}"
      
      if self.lista_fases.index(self.ind0[0]) < self.lista_fases.index(self.ind0[1]):
        puntos = self.intsc_prob_goles(X, model, alterno=1)
      else:
        puntos = self.intsc_prob_goles(X, model, alterno=0)

      for points_ in puntos:
        home, away = df_fixture_updated["home"], df_fixture_updated["away"]
        point_h, point_a = points_[0][0], points_[1][0]

        if point_h > point_a:
          ganador = home
          perdedor = away
        else:
          ganador = away
          perdedor = home

        self.dic_matches[partido]["ganador"] = ganador
        self.dic_matches[partido]["perdedor"] = perdedor

        Y0 = np.append(Y0, [point_h, point_a])

      df_fixture_updated["score_0"], df_fixture_updated["score_1"] = Y0[0::2], Y0[1::2]
      df_fixture_updated = calculo_metricas_1(df_fixture_updated)

    homes, aways, Winners, Losers = [], [], [], []
    for linea in df_fixture_updated.itertuples():
      indice = (self.n_partidos - tamagno) + linea.Index
      partido = f"partido {indice}"

      homes.append(self.id_pais[linea.home])
      aways.append(self.id_pais[linea.away])
      Winners.append(self.id_pais[self.dic_matches[partido]["ganador"]])
      Losers.append(self.id_pais[self.dic_matches[partido]["perdedor"]])
    
    df_nombres = pd.DataFrame(data={"home": homes, "away": aways, "Winner": Winners, "Loser": Losers}, index=range(len(homes)))
    print(df_nombres)

    return df_fixture_updated

  def seleccionar(self, df_fixture_, equipo, ind0):
    """
    ¿QUE HACE?
    Entrega las ultiimas metricas de rendimiento registradas, ya sea por fase de grupos o por alguno de los rounds de la fase final en el parámetro (df_fixture_), para el equipo de interés entregado en el parámetro (equipo), ordenando la respuesta segun se indique el el parámetro (ind0)

    ¿COMO LO HACE?
    0-) se definen unas variables (indice_0, indice_1) que poseen los indices de las filas del df (df_fixture_) donde esté el (equipo), ya sea como home o away

    1-) se define una variable (mas_g) con el index máximo entre (indice_0, indice_1)

    2-) se tienen 2 condicionles para actuar segun si el index máximo está entre los partidos jugados como home o away

      2.0) si el máximo está en la lista de partidos home: se toman dentro de una variable (seleccion) las primeras 10 columnas (0-9) con la informacion del rendimiento acumulada ahi (home, PTS_0, PJ_0, PG_0, PP_0, PE_0, GF_0, GC_0, D_0, score_0), se sustituye el nombre home por away si (ind0) dice away y se entrega (seleccion)

      2.1) si el máximo está en la lista de partidos away: se toman dentro de una variable (seleccion) las ultimas 10 columnas (10-19) con la informacion del rendimiento acumulada ahi (away, PTS_1, PJ_1, PG_1, PP_1, PE_1, GF_1, GC_1, D_1, score_1), se sustituye el nombre away por home si (ind0) dice home y se entrega (seleccion)

    Args:

      - df_fixture_: será el df (df_encuentro_jugado) respecto al 1er round de la fase final o 
      a la 2da fase de grupos

      - equipo: será el nombre del equipo a buscar dentro de los partidos jugados en 
      (df_fixture_)

      - ind0: indica si la respuesta debe ser dirigida a un partido como home o away
    
    Returns:

      - seleccion: Serie contenedora de la informacion de rendimiento registrada para el ultimo partido jugado por (equipo) en (df_fixture_), formateada segun el lugar que tomará el (equipo) en la próxima fase "home o away"
    """
    indice_0 = df_fixture_[df_fixture_.home == equipo].index
    indice_1 = df_fixture_[df_fixture_.away == equipo].index
    
    mas_g = max(indice_0.tolist() + indice_1.tolist())
    
    cols_per_team = 1 + len(ALL_METRICS) + 1

    if mas_g in indice_0:
      seleccion = df_fixture_.iloc[mas_g, :cols_per_team]
      if ind0 == "away":
        seleccion.rename(index={"home": "away"}, inplace=True)
      return seleccion

    elif mas_g in indice_1:
      seleccion = df_fixture_.iloc[mas_g, cols_per_team:]
      if ind0 == "home":
        seleccion.rename(index={"away": "home"}, inplace=True)
      return seleccion

  def camb_grp_elm_smpl(self, df_encuentro_jugado, df_encuentro_siguiente, dt, year=None):
    """
    ¿QUE HACE?
    Organiza a los equipos  que jugaran el primer round de la fase final (generalmente un knockout u octavos de final en un torneo de eliminacion simple) tras acabar la fase de grupos y haberse definido los 1ros a 4tos lugares por grupo dentro de la misma, por su acumulacion de puntos
    
    ¿COMO LO HACE?
    0-) Se almacenann las columnas de la fase de grupos (df_encuentro_jugado), se crea un diccionario (dic) con claves lugar obtenido en fase de grupos y valor numero correspondiente a dicho lugar (ej. "Third-p":2, en referencia a el index en df para ese equipo). Un diccionario (dic_f20) vacio par el caso de tener una 2da fase de grupos, la dimension del df de el proximo round (tamagno_0, el 1ro de la fase final), el numero de columnas (tamagno_1) para el df de el proximo round, finalmente, el df vacio para generar el correspondiente al 1er round de la fase final a disputarse (dffix)

    1-) Se definen 2 condicionales para los casos en que se jugará una fase de grupos (para mundiales pasados esto solo ocurre entre 1974 y 1982, de darse de nuevo este formato en el futuro, debe tenerse al menos el nombre y cantidad de los grupos para esta 2da fase). Lo que se hace es agrupar en variables, el contenido del df (df_encuentro_siguiente) correspondiente a los partidos de esa 2da fase, teniendo cada variable el numero de partidos por grupo para esta fase. Se almacenan todos en un diccionario que debera retornarse al final de la funcion con claves nombre de los grupos y valor variables con los partidos por grupos

    2-) Se define un condicional para trabajar con rounds de mas de 1 partido, o de 1 solo partido. para el 1er caso
      
      2.0-) se iteran en forma de tupla los partidos del df_encuentro_siguiente, se almacena de cada partido iterado el equipo local (home) y visitante (away), se definen unas variables vacias para su uso futuro (equipo0, equipo1)

        2.0.0) para el caso de tener un df_encuentro_siguiente de 3 columnas, se busca: el índice correspondiente a la posición por puntaje alcanzada por el equipo home y away, según el (dic) en fase de grupos (ubicacion0, ubicacion1), el grupo correspondiente al equipo home y away que juegan ese partido iterado (g0, g1), la búsqueda del nombre del equipo, según la posición alcanzada dentro del grupo definido (equipo0, equipo1), mediante la funcion "loc"

        2.0.1) Para el caso de tener 2 o 4 columnas (entrada del usuario) (equipo0, equipo1) serán iguales a los (home, away) que se iteran del df (df_encuentro_siguiente)
      
      Luego se usa el metodo (seleccionar) para obtener las metricas de rendimiento para los (equipo0, equipo1) segun el df (df_encuentro_jugado) y se almacenan estos valores en un df (rndTeam0 y rndTeam1) que luego se concatena como columnas a un df (partido) y este a su vez se concatena como fila a un (dffix)

      2.1-) se almacena en unas variables (home0, away0) los nombres de los equipos home y away que juegan la siguiete round (el 1ro tras la fase de grupos)
      
        2.1.0-) para el caso de tener un df_encuentro_siguiente de 3 columnas, se almacena en unas variables (home0, away0) los nombres de los equipos home y away que juegan la siguiete round (el 1ro tras la fase de grupos). Se obtiene el lugar e indice de fila en el df del grupo para los equipos home y away mediante la funcion "split" para el home y el away, mediante el (dic) y se les almacena en las variables (ubicacion0, ubicacion1). Luego se crea el nombre del grupo a buscar en (g0, g1), luego se busca el df de ese grupo en el (dt) y se obtiene el nombre de país correspondiente al lugar del home o away mediante la función "loc" (equipo0, equipo1)

        2.1.1-) para el caso de tener un df_encuentro_siguiente de 2 o 4 columnas (entrada del usuario) (equipo0, equipo1) serán iguales a los (home, away) que se obtienen del df (df_encuentro_siguiente)
      
      Luego se usa el metodo (seleccionar) para obtener las metricas de rendimiento para los (equipo0, equipo1) segun el df (df_encuentro_jugado) y se almacenan estos valores en unos df (rndTeam0 y rndTeam1) que luego se concatena como columnas a un df (partido) y este a su vez se concatena como fila a un df (dffix)
    
    3-) se definen las columans para (dffix) y se depuran los indices

    4-) se divide la respuesta final entre 2 condicionales

      4.0) el año del mundial en que se predice es tal que se presenta una 2da fase de grupos (1974, 1978, 1982)

        4.0.0-) Si la dimension del df (df_encuentro_siguiente) es propia de entradas historicas (3) o si no se esta entre los estado actual y objetivo (grupo, doce); se define un diccionario vacio para recepción (dic_f21), se itera el contenido de los puntajes por grupo de la primera fase de grupos en (dic_f20), se define un df para recepción (new_df), se iteran las filas del df de desempeño por grupo en la primera fase de grupos, se deja en una variable (ef2) el desempeño del grupo definido para los índices correspondientes al grupo dejado en la columna (pais) de la fila iterada del (dic_f20) y la posición del equipo de interés presente hasta los últimos 8 índices de la entrada (pais) de la fila iterada según el índice marcado por (dic), se concatena a (new_df) el contenido de (ef2), se almacena (new_df) en el grupo iterado desde (dic_f20) dentro del diccionario de recepcion (dic_f21), finalmente se retornan el df (dffix) con el orden e informacion para los partidos del siguiente round y el diccionario (dic_f21)
    
        4.0.1-) Si la dimensión del df (df_encuentro_siguiente) es propia de entradas del usuario (2 o 4), se iteran los grupos del diccionario (dic_f20), se genera un df y una variable para almacen de puntajes (punto), se iteran los equipos de la columna (pais) del grupo iterado, se iteran los grupos del diccionario de la fase de grupos previa y se actualiza la variable de (punto) con el puntaje del equipo iterado en el grupo de la fase previa donde este se ubicó, se actualiza la variable (punto) a un df con 2 filas (pais, punto), se concatena como fila en un df (df0) el ahora array (punto), se almacena en ese (dic_f20) el grupo iterado de la nueva fase y el (df0) con los (pais, punto) actualizados, finalmente, se entregan el df (dffix) junto con el diccionario (dic_f20)
      
      4.1) se esta en un mundial que no incluye una 2da fase de grupos. La respuesta es solo el df (dffix) con el orden e informacion para los partidos del siguiente round

    Args:
    
      - df_encuentro_jugado: df con el orden e informacion del rendimiento de cada equipo de la fase de grupos o fase previa al 1er round de la fase final

      - df_encuentro_siguiente: df con el orden y nombres de equipos o ids de busqueda para los equipos que disputaran el 1er round de la fase final, tras la fase de grupos

      - dt: diccionario con informacion del rendimiento acumulado durante la disputa de la fase de grupos, organizada en clave: nombre del grupo (Grupo X) y valor: df de 4 filas y columnas "pais, puntaje"

      - year: define si el año del mundial en estudio es de los que han tenido una 2da fase de grupos

    Returns:

      - (dffix, dic_f21): el primero es el df con el orden e informacion de los equipos que disputarán el siguiente round. el segundo es el diccionario actualizado desde las claves de busqueda de partidos de la 1ra fase de grupos para los equipos de cada grupo con la informacion de desempeño acumulada en esa fase
    
      - (dffix, dic_f20): el primero es el df con el orden e informacion de los equipos que disputarán el siguiente round. el segundo es el diccionario actualizado segun la informacion del usuario, para la disputa de la 2da fase de grupos

      - (dffix): para los mundiales con solo una fase de grupos
    """
    columnas = df_encuentro_jugado.columns.to_list()
    dic = {"Winner": 0, "Runners-up": 1, "Third-p": 2, "Fourth-p": 3}
    dic_f20 = {}
    tamagno_0 = len(df_encuentro_siguiente.shape)
    tamagno_1 = df_encuentro_siguiente.shape[-1]
    dffix = pd.DataFrame()

    if year in (1974, 1978):
      uno = pd.concat([df_encuentro_siguiente.home.iloc[:6], df_encuentro_siguiente.away.iloc[:6]], axis=0).unique().tolist()
      dos = pd.concat([df_encuentro_siguiente.home.iloc[6:12], df_encuentro_siguiente.away.iloc[6:12]], axis=0).unique().tolist()
      uno = pd.DataFrame(data=np.array(uno), columns=["pais"]).reset_index(drop=True)
      dos = pd.DataFrame(data=np.array(dos), columns=["pais"]).reset_index(drop=True)
      dic_f20 = {"Grupo A": uno, "Grupo B": dos}
    
    elif year in (1982,):
      uno = pd.concat([df_encuentro_siguiente.home.iloc[:3], df_encuentro_siguiente.away.iloc[:3]], axis=0).unique().tolist()
      dos = pd.concat([df_encuentro_siguiente.home.iloc[3:6], df_encuentro_siguiente.away.iloc[3:6]], axis=0).unique().tolist()
      tres = pd.concat([df_encuentro_siguiente.home.iloc[6:9], df_encuentro_siguiente.away.iloc[6:9]], axis=0).unique().tolist()
      cuatro = pd.concat([df_encuentro_siguiente.home.iloc[9:12], df_encuentro_siguiente.away.iloc[9:12]], axis=0).unique().tolist()

      uno = pd.DataFrame(data=np.array(uno), columns=["pais"]).reset_index(drop=True)
      dos = pd.DataFrame(data=np.array(dos), columns=["pais"]).reset_index(drop=True)
      tres = pd.DataFrame(data=np.array(tres), columns=["pais"]).reset_index(drop=True)
      cuatro = pd.DataFrame(data=np.array(cuatro), columns=["pais"]).reset_index(drop=True)

      dic_f20 = {"Grupo A": uno,
                 "Grupo B": dos,
                 "Grupo C": tres,
                 "Grupo D": cuatro}

    if tamagno_0 > 1:
      for row_fix in df_encuentro_siguiente.itertuples():
        home, away = row_fix.home, row_fix.away
        equipo0, equipo1 = None, None

        if tamagno_1 == 3:
          ubicacion0, ubicacion1 = dic[home[:home.find(" ")]], dic[away[:away.find(" ")]]
          g0, g1 = home[home.find(" ")+1:], away[away.find(" ")+1:]
          equipo0, equipo1 = dt[g0].loc[ubicacion0, "pais"], dt[g1].loc[ubicacion1, "pais"]

        elif (tamagno_1 == 2 or tamagno_1 == 4):
          equipo0, equipo1 = home, away

        rndTeam0 = self.seleccionar(df_encuentro_jugado, equipo0, "home")
        rndTeam0 = pd.DataFrame(data=[rndTeam0.values], columns=rndTeam0.index).reset_index(drop=True)

        rndTeam1 = self.seleccionar(df_encuentro_jugado, equipo1, "away")
        rndTeam1 = pd.DataFrame(data=[rndTeam1.values], columns=rndTeam1.index).reset_index(drop=True)

        partido = pd.concat([rndTeam0, rndTeam1], axis=1, ignore_index=True)
        dffix = pd.concat([dffix, partido], axis=0)

    elif tamagno_0 == 1:
      home0, away0 = df_encuentro_siguiente.home, df_encuentro_siguiente.away
      if tamagno_1 == 3:
        home1, away1 = home0.split(" "), away0.split(" ")
        ubicacion0, ubicacion1 = dic[home1[0]], dic[away1[0]]
        g0, g1 = home1[1]+" "+home1[2], away1[1]+" "+away1[2]
        equipo0, equipo1 = dt[g0].loc[ubicacion0, "pais"], dt[g1].loc[ubicacion1, "pais"]

      elif (tamagno_1 == 2 or tamagno_1 == 4):
        equipo0, equipo1 = home0, away0

      rndTeam0 = self.seleccionar(df_encuentro_jugado, equipo0, "home")
      rndTeam0 = pd.DataFrame(data=[rndTeam0.values], columns=rndTeam0.index).reset_index(drop=True)
      
      rndTeam1 = self.seleccionar(df_encuentro_jugado, equipo1, "away")
      rndTeam1 = pd.DataFrame(data=[rndTeam1.values], columns=rndTeam1.index).reset_index(drop=True)
      
      partido = pd.concat([rndTeam0, rndTeam1], axis=1, ignore_index=True)
      dffix = pd.concat([dffix, partido], axis=0)

    dffix.columns = columnas
    dffix = dffix.reset_index(drop=True)

    if year in (1974, 1978, 1982):
      dic_f21 = {}
      if (not self.ind0[0] == "grupo" and not self.ind0[1] == "doce") or (tamagno_1 == 3):
        for des0 in dic_f20.items():
          new_df = pd.DataFrame(columns=("pais", "Pts"))
          for des1 in des0[1].itertuples():
            ef2 = dt[des1.pais[-7:]].iloc[dic[des1.pais[:-8]]]
            new_df = pd.concat([new_df, pd.DataFrame([ef2], columns=new_df.columns)], axis=0)
          dic_f21[des0[0]] = new_df
        return dffix, dic_f21

      else:
        for des0 in dic_f20.items():
          df0 = pd.DataFrame()
          punto = 0
          for des00 in des0[1].pais.values:
            for des01 in dt.keys():
              if des00 in dt[des01].pais.values:
                punto = dt[des01][dt[des01].pais == des00]["Pts"].values[0]

            punto = pd.DataFrame(data=np.array([[des00, punto]]), columns=["pais", "Pts"])
            df0 = pd.concat([df0, punto], axis=0)
          dic_f20[des0[0]] = df0
        return dffix, dic_f20

    else:
      return dffix

  def ordenar_elm_smpl(self, df_encuentro_jugado, df_encuentro_siguiente, tipo="Winner"):
    """
    ¿QUE HACE?
    Organiza los equipos que jugarán los partidos de cada round de la fase final desde el primero hasta el ultimo, valiendose de los nombres presentes en la seccion del fixture entregada como(df_encuentro_siguiente) para obtener a los equipos desde el df (df_encuentro_jugado), según las victorias y derrotas por partido, marcadas en (dic_matches), dejando en un df (df_encuentro_siguiente) nuevo con el orden y la informacion de rendimiento necesaria para la inferencia en la funcion (get_winner)

    ¿COMO LO HACE?
    0-) se define un df (dffix) vacio para la recepcion del orden de juego y rendimiento de cada equipo, una variable (columnas) con el nombre de las columnas del df (df_encuentro_jugado), una variable (tamagno_1) con el número de columnas en el df (df_encuentro_siguiente) para determinar si se organiza hacia una prediccion del usuario o tras esta
    
    1-) Se tienen 2 condicionales para dirigir el ordenamiento de partidos segun el numero de partidos del siguiente round
      
      1.0) Para cuando (tipo) sea "Winner" y no sea la final (1 solo partido por round). Se iteran los nombres de los equipos que jugarán como home y away en el próximo round df (df_encuentro_siguiente)

        1.0.0) se tienen 2 condicionales

          1.0.0.0) para cuando el numero de columnas del df del siguiente round sea 3. En este caso home y away corresponderan a claves de la forma (Winner-Looser Partido X) que se almacenan por separado en unas variables (equipo0, equipo1), con estas se busca al ganador registrado para dicha clave buscando en el diccionario (dic_matches) con la clave de nombre del partido que redirige a otro diccionario cuya clave corresponde al parámetro (tipo)

          1.0.0.1) para cuando el numero de columnas del df del siguiente round sea 2 o 4. En este caso lo entregado corresponderá a una consulta del usuario, por lo que home y away corresponderán a los nombres directamente de los equipos que disputan el proximo round y no la clave mostrada para el condicional anterior, dichos nombres se almacenan en las variables (equipo0, equipo1)

        1.0.1) Se usa el metodo (seleccionar) para obtener las metricas de rendimiento para los (equipo0, equipo1) segun el df (df_encuentro_jugado) y se almacenan estos valores en unos df (rndTeam0 y rndTeam1) que luego se concatena como columnas a un df (partido) y este a su vez se concatena como fila a un df (dffix)

      1.1) para cuando (tipo) sea "Looser" o sea la final (1 solo partido por round). Se toman las columnas home y away del df (df_encuentro_siguiente) y se almacenan en unas variables (home, away)

        1.1.0) se tienen 2 condicionales

          1.1.0.0) cuando el numero de columnas del df (df_encuentro_siguiente) es 3, se tiene una estructura de home y away (Winner-Looser Partido X) y se busca en el diccionario (dic_matches) con clave (Partido X) redirigiéndose a un diccionario con clave (tipo), almacenándose el resultado del ganador o perdedor del partido referenciado en unas variables (equipo0, equipo1)

          1.1.0.1) cuando el numero de columnas del df (df_encuentro_siguiente) es 2 o 4,  home y away son solo los nombres de los equipos que, segun la inferencia buscada por el usuario, disputarian el siguiente round, almacenándose el resultado del ganador o perdedor del partido referenciado en unas variables (equipo0, equipo1)
        
        1.1.1) Se usa el metodo (seleccionar) para obtener las metricas de rendimiento para los (equipo0, equipo1) segun el df (df_encuentro_jugado) y se almacenan estos valores en unos df (rndTeam0 y rndTeam1) que luego se concatena como columnas a un df (partido) y este a su vez se concatena como fila a un df (dffix)

    2-) se resetean los indices del df (dffix) y se le adaptan las columnas de la variable (columnas) correspondientes al df (df_encuentro_jugado)

    Args:

      - df_encuentro_jugado: df con la informacion actualizada para cada equipo que disputara el round previo y que jugará en el proximo
      
      - df_encuentro_siguiente: df contenedor de los partidos que habrán de jugarse en el siguiente round, con el nombre de los equipos definidos por una consulta del usuario (2 o 4 columnas) o las claves para buscar a los equipos que disputarán el siguiente round de la forma (Winner-Looser Partido X) (3 columnas)
      
      - tipo: el resultado del partido previo que define el lugar disputado en el siguiente round

    Return:

      - dffix: df contenedor del orden de juego del siguiente round, con los equipos que lo disputaran y su informacion de rendimiento

    """
    dffix = pd.DataFrame()
    columnas = df_encuentro_jugado.columns.to_list()
    tamagno_1 = df_encuentro_siguiente.shape[-1]

    if tipo == "Winner" and not len(df_encuentro_siguiente.shape) == 1:
      for home, away in zip(df_encuentro_siguiente["home"], df_encuentro_siguiente["away"]):
        if tamagno_1 == 3:
          equipo0 = self.dic_matches[home[len(tipo)+1:]][tipo]
          equipo1 = self.dic_matches[away[len(tipo)+1:]][tipo]
        elif (tamagno_1 == 2 or tamagno_1 == 4):
          equipo0, equipo1 = home, away

        rndTeam0 = self.seleccionar(df_encuentro_jugado.iloc[:, :-3], equipo0, "home") 
        rndTeam0 = pd.DataFrame(data=[rndTeam0.values], columns=rndTeam0.index).reset_index(drop=True)
        
        rndTeam1 = self.seleccionar(df_encuentro_jugado.iloc[:, :-3], equipo1, "away")
        rndTeam1 = pd.DataFrame(data=[rndTeam1.values], columns=rndTeam1.index).reset_index(drop=True)
        
        partido = pd.concat([rndTeam0, rndTeam1], axis=1, ignore_index=True)
        dffix = pd.concat([dffix, partido], axis=0)

    elif tipo == "Loser" or len(df_encuentro_siguiente.shape) == 1:
      home, away = df_encuentro_siguiente["home"], df_encuentro_siguiente["away"]

      if tamagno_1 == 3:
        equipo0 = self.dic_matches[home[len(tipo) + 1:]][tipo]
        equipo1 = self.dic_matches[away[len(tipo) + 1:]][tipo]
      elif (tamagno_1 == 2 or tamagno_1 == 4):
        equipo0, equipo1 = home.item(), away.item()

      rndTeam0 = self.seleccionar(df_encuentro_jugado.iloc[:, :-3], equipo0, "home")
      rndTeam0 = pd.DataFrame(data=[rndTeam0.values], columns=rndTeam0.index).reset_index(drop=True)

      rndTeam1 = self.seleccionar(df_encuentro_jugado.iloc[:, :-3], equipo1, "away")
      rndTeam1 = pd.DataFrame(data=[rndTeam1.values], columns=rndTeam1.index).reset_index(drop=True)

      partido = pd.concat([rndTeam0, rndTeam1], axis=1, ignore_index=True)
      dffix = pd.concat([dffix, partido], axis=0)

    dffix = dffix.reset_index(drop=True)
    dffix.columns = columnas

    return dffix

# Global setups
clasificaciones = formar_dataset_real("clasificaciones")
agnos = [1934, 1938, 1950, 1954, 1958, 1962, 1966, 1970, 1974, 1978, 1982, 1986, 1990, 1994, 1998, 2002, 2006, 2010, 2014, 2018, 2022]
dic_agnos = {}

for des0 in agnos:
  df_partidos_agnos = pd.DataFrame(columns=["home", "score_0", "score_1", "away", "agno"])

  for des1 in set(clasificaciones["agno"].tolist()):
    if des0 > agnos[0]:
      if (int(des1) <= des0 and int(des1) > agnos[agnos.index(des0)-1]):
        eliminatoria = clasificaciones.loc[clasificaciones["agno"] == des1][["home", "score_0", "score_1", "away", "agno"]]
        df_partidos_agnos = pd.concat([df_partidos_agnos, eliminatoria], axis=0)

    elif not (des0 > agnos[0]):
      if int(des1) <= des0:
        eliminatoria = clasificaciones.loc[clasificaciones["agno"] == des1][["home", "score_0", "score_1", "away", "agno"]]
        df_partidos_agnos = pd.concat([df_partidos_agnos, eliminatoria], axis=0)

  eliminatorias = pd.DataFrame()
  for idx, pais in enumerate(set(df_partidos_agnos.home.tolist() + df_partidos_agnos.away.tolist())):
    eliminatorias = pd.concat([eliminatorias, funcion_tabla_desempegno(df_partidos_agnos, pais, idx, des0)], axis=0)

  dic_agnos[str(des0)] = eliminatorias

def consulta_general(pregunta_0, df5):
  """
  ¿QUE HACE?
  Realiza la predicción de resultados para un mundial específico, creando el objeto de la clase (func_prediccion_orden) necesario segun el año al que quiere aplicarse inferencia, generando los datasets necesarios, generando parametros para calcular la fuerza aproximada de cada equipo, cargando el modelo entrenado y aplicando las funciones de predicción para obtener los resultados esperados

  ¿COMO LO HACE?

  0-) Se tienen 2 condicionales para definir el objeto (objs) de la clase (func_prediccion_orden) segun el año del mundial a predecir:

    0.0-) Si el año no está entre (1986, 1990 o 1994):
      
      0.0.0-) Se crea el objeto sin considerar el parametro (grupos_mundiales), colocando los parametros de clase convencionales:

        - ind_partidos: cadena con la forma ("partidos") para indicar la busqueda correcta en DB con el ORM de django

        - ind_fixtures: cadena contenedora del año del mundial de interes para indicar la busqueda correcta de su fixture en DB con el ORM de django
      
        - df_prediccion_usuario: df de consulta del usuario para hacer inferencias personalizadas. Idealmente se incluye o no en la salida del metodo (create_features) la ---modalidad de inferencia---
      
        - clsif: df con las clasificaciones historicas de mundiales para obtener el total absoluto de equipos participantes en el mundial a predecir
    
        - obj0: un entero con el año del mundial a predecir, para indicar si el componente de indice 0 de la salida del metodo (create_features) incluye (fase final fixture del mundial jugado, df consulta del usuario) o (contenido del mundial jugado) 

    0.1-) Si el año está entre (1986, 1990 o 1994):
      
      0.1.0-) Se define al parametro (grupos_mundiales) como "grupos" y se crea el objeto considerando este parametro, colocando unos parametros de clase extra: 

        - (ind_grupos_mundiales): (sera una cadena contenedora de la palabra "grupos" para permitir el uso del array (self.dic_emp_0)

        - (agno): un entero que indica el año del mundial con formato de emparejamiento por mejores terceros, para referenciar en el atributo de instancia diccionario (self.dic_emp_1) con pares clave (1986, 1990, 1994)-valor (tupla con dos subtuplas internas, cada una con el momento de juego para cada partido fijo en la 1ra y variable en la 2da)

  1-) Se llama al metodo (create_features) del objeto (objs) usando como argumento el (dic_agnos) para generar los datasets necesarios para la predicción, almacenándolos en las variables (df1, df2, df3):

  2- Se crean atributos:

    - df1: (df conenedor de la fase final fixture del mundial jugado, df consulta del usuario) o (contenido del mundial jugado)
  
    - df2: (df conenedor de la fase de grupos del fixture)
    
    - df3: (df conenedor de la fase final del fixture)

  3-) Se crea la variable (agg0) como un df que concatena en filas y reiniciando el índice con el parametro (ignore_index=True), las columnas (home) de (df2) y (df1[0]), renombrando la columna resultante como (Team) mediante el metodo (rename)

  4-) Se crea la variable (pts0) como un df que concatena en filas y reiniciando el índice con el parametro (ignore_index=True), las columnas (score_0) de (df2) y (df1[0]), renombrando la columna resultante como (goles) mediante el metodo (rename)

  5-) Se concatena (agg0) y (pts0) en columnas para generar un df (agg0) con las columnas (Team, goles), luego se agrupa por (Team) y se calcula el promedio de goles por equipo mediante el metodo (groupby) y (mean)

  6-) Se crea la variable (agg1) como un df que concatena en filas y reiniciando el índice con el parametro (ignore_index=True), las columnas (away) de (df2) y (df1[0]), renombrando la columna resultante como (Team) mediante el metodo (rename)

  7-) Se crea la variable (pts1) como un df que concatena en filas y reiniciando el índice con el parametro (ignore_index=True), las columnas (score_1) de (df2) y (df1[0]), renombrando la columna resultante como (goles) mediante el metodo (rename)
  
  8-) Se crea la variable (pts1) como un df que concatena en filas y reiniciando el índice con el parametro (ignore_index=True), las columnas (score_1) de (df2) y (df1[0]), renombrando la columna resultante como (goles) mediante el metodo (rename)

  9-) Se concatena (agg1) y (pts1) en columnas para generar un df (agg1) con las columnas (Team, goles), luego se agrupa por (Team) y se calcula el promedio de goles por equipo mediante el metodo (groupby) y (mean)

  10-) Se actualiza el diccionario (dic_fuerza) del objeto (objs) con  llaves (home) y (away) respectivamente y como claves los df (agg0) y (agg1)

  11-) Se define la variable (dir_modelo) con la ruta del mejor modelo entrenado disponible

  12-) Se carga el modelo entrenado mediante la función (tf.keras.models.load_model) y se almacena en la variable (modelo)

  13-) Se crea una variable (dict_g) llamando al medtodo (grupos_anio_interes) del objeto (objs) con parametro igual al año del mundial de interes y se obtiene un diccionario con claves de cada grupo que jugó ese mundial ("Grupo A", "Grupo B", pj) y valor un df contenedor de los equipos del grupo, parametros de rendimiento seteados a 0, el grupo mismo y el año de juego

  14-) Se crean unos condicionales para definir si el mundial a predecir esta entre alguno de los periodos en que el formato del torneo tuvo algun cambio significativo:

    - df_ent: df con los resultados predichos para cada fase del mundial de interes y nombres de equipos ya en formato texto, con las alteraciones que el usuario haya hecho en su consulta, concatenando los dfs de cada fase del mundial de interes (fase de grupos, 2da fase de grupos, semifinales, partido por el tercer lugar y final) cada uno con sus resultados predichos y nombres de equipos ya en formato texto

  """

  if not pregunta_0 in ("1986", "1990", "1994"):
    objs = func_prediccion_orden("partidos", pregunta_0[-4:], df5, clasificaciones, obj0=int(pregunta_0[-4:]))
  else:
    grupos_mundiales = "grupos"
    objs = func_prediccion_orden("partidos", pregunta_0[-4:], df5, clasificaciones, ind_grupos_mundiales=grupos_mundiales, agno=int(pregunta_0[-4:]), obj0=int(pregunta_0[-4:]))

  df_features = objs.create_features(dic_agnos)
  df1, df2, df3 = df_features[0], df_features[1], df_features[2]

  print("primero: (df1)")
  print(df1)
  print("")
  print("")
  print("segundo: (df2)")
  print(df2)
  print("")
  print("")
  print("tercero: (df3)")
  print(df3)

  agg0 = pd.concat([df2.home, df1[0].home], axis=0, ignore_index=True)
  agg0 = agg0.rename(columns={"home": "Team"})

  pts0 = pd.concat([df2[["score_0"]], df1[0][["score_0"]]], axis=0, ignore_index=True)
  pts0 = pts0.rename({"score_0": "goles"})

  agg0 = pd.concat([agg0, pts0], axis=1)
  agg0 = agg0.groupby(["Team"]).mean()

  agg1 = pd.concat([df2.away, df1[0].away], axis=0, ignore_index=True)
  agg1 = agg1.rename(columns={"away": "Team"})

  pts1 = pd.concat([df2[["score_1"]], df1[0][["score_1"]]], axis=0, ignore_index=True)
  pts1 = pts1.rename({"score_1": "goles"})

  agg1 = pd.concat([agg1, pts1], axis=1)
  agg1 = agg1.groupby(["Team"]).mean()

  objs.dic_fuerza["home"] = agg0
  objs.dic_fuerza["away"] = agg1

  dir_modelo = "C:\\Users\\Usuario\\Tareas\\p_mundiales\\programacion\\entregas\\proyecto_1\\data\\modelos\\modelo-LSTM-(series_temporales)-16-12-2024-0.h5"
  modelo = tf.keras.models.load_model(dir_modelo)
  
  dict_g = objs.grupos_anio_interes(pregunta_0[-4:])

  if "1998" in pregunta_0 or "2002" in pregunta_0 or "2006" in pregunta_0 or "2010" in pregunta_0 or "2014" in pregunta_0 or "2018" in pregunta_0 or "2022" in pregunta_0:
    df_fixture_group_, df_fixture_knockout, df_fixture_quarter, df_fixture_semi, df_fixture_third, df_fixture_final = None, None, None, None, None, None

    if "grupo" in pregunta_0:
      objs.ind0[1] = "grupo"
      df_fixture_group_ = agregar_features(df1[1].copy(), dic_agnos, ind=0)
      df_fixture_knockout = df3[["home", "score", "away"]].iloc[0:8].copy()
      df_fixture_quarter = df3[["home", "score", "away"]].iloc[8:12].copy()
      df_fixture_semi = df3[["home", "score", "away"]].iloc[12:14].copy()
      df_fixture_third = df3[["home", "score", "away"]].iloc[14].copy()
      df_fixture_final = df3[["home", "score", "away"]].iloc[15].copy()

    elif "knockout" in pregunta_0:
      objs.ind0[1] = "knockout"
      df_fixture_group_ = df2.copy()
      df_fixture_knockout = df1[1].copy()
      df_fixture_quarter = df3[["home", "score", "away"]].iloc[8:12].copy()
      df_fixture_semi = df3[["home", "score", "away"]].iloc[12:14].copy()
      df_fixture_third = df3[["home", "score", "away"]].iloc[14].copy()
      df_fixture_final = df3[["home", "score", "away"]].iloc[15].copy()

    elif "quarter" in pregunta_0:
      objs.ind0[1] = "quarter"
      df_fixture_group_ = df2.copy()
      df_fixture_knockout = df1[0][["home", "away"]].iloc[0:8].copy()
      df_fixture_quarter = df1[1].copy()
      df_fixture_semi = df3[["home", "score", "away"]].iloc[12:14].copy()
      df_fixture_third = df3[["home", "score", "away"]].iloc[14].copy()
      df_fixture_final = df3[["home", "score", "away"]].iloc[15].copy()

    elif "semi" in pregunta_0:
      objs.ind0[1] = "semi"
      df_fixture_group_ = df2.copy()
      df_fixture_knockout = df1[0][["home", "away"]].iloc[0:8].copy()
      df_fixture_quarter = df1[0][["home", "away"]].iloc[8:12].copy()
      df_fixture_semi = df1[1].copy()
      df_fixture_third = df3[["home", "score", "away"]].iloc[14].copy()
      df_fixture_final = df3[["home", "score", "away"]].iloc[15].copy()

    elif "third" in pregunta_0:
      objs.ind0[1] = "third"
      df_fixture_group_ = df2.copy()
      df_fixture_knockout = df1[0][["home", "away"]].iloc[0:8].copy()
      df_fixture_quarter = df1[0][["home", "away"]].iloc[8:12].copy()
      df_fixture_semi = df1[0][["home", "away"]].iloc[12:14].copy()
      df_fixture_third = df1[1].copy()
      df_fixture_final = df3[["home", "score", "away"]].iloc[15].copy()

    elif "final" in pregunta_0:
      objs.ind0[1] = "final"
      df_fixture_group_ = df2.copy()
      df_fixture_knockout = df1[0][["home", "away"]].iloc[0:8].copy()
      df_fixture_quarter = df1[0][["home", "away"]].iloc[8:12].copy()
      df_fixture_semi = df1[0][["home", "away"]].iloc[12:14].copy()
      df_fixture_third = df1[0][["home", "away"]].iloc[14].copy()
      df_fixture_final = df1[1].copy()

    objs.ind0[0] = "grupo"
    dict_table = objs.fase_de_grupos(dict_g, df_fixture_group_, modelo) 

    objs.n_partidos = df_fixture_group_.shape[0] + df_fixture_knockout.shape[0] + 1
    df_fixture_knockout = objs.camb_grp_elm_smpl(df_fixture_group_, df_fixture_knockout, dict_table) 

    objs.ind0[0] = "knockout"
    objs.get_winner(df_fixture_knockout, modelo) 

    print("indice base", objs.n_partidos)
    df_fixture_quarter = objs.ordenar_elm_smpl(df_fixture_knockout, df_fixture_quarter) 

    objs.ind0[0] = "quarter"
    objs.get_winner(df_fixture_quarter, modelo)

    print("indice base", objs.n_partidos)
    df_fixture_semi = objs.ordenar_elm_smpl(df_fixture_quarter, df_fixture_semi)

    objs.ind0[0] = "semi"
    objs.get_winner(df_fixture_semi, modelo)

    df_fixture_third = objs.ordenar_elm_smpl(df_fixture_semi, df_fixture_third, tipo="Loser")

    objs.ind0[0] = "third"
    objs.get_winner(df_fixture_third, modelo)

    df_fixture_final = objs.ordenar_elm_smpl(df_fixture_semi, df_fixture_final)

    print(f"\n\n----------\n----------\n\nFINAL - FINAL - FINAL: --- {pregunta_0} ---\n----------\n----------\n")
    objs.ind0[0] = "final"
    objs.get_winner(df_fixture_final, modelo)

    df_ent = pd.concat([df_fixture_group_[["home", "score_0", "score_1", "away"]], df_fixture_knockout[["home", "score_0", "score_1", "away"]], df_fixture_quarter[["home", "score_0", "score_1", "away"]], df_fixture_semi[["home", "score_0", "score_1", "away"]], df_fixture_third[["home", "score_0", "score_1", "away"]], df_fixture_final[["home", "score_0", "score_1", "away"]]], axis=0)

    df_ent.home = df_ent.home.map(objs.id_pais)
    df_ent.away = df_ent.away.map(objs.id_pais)

    return df_ent

  elif "1934" in pregunta_0 or "1938" in pregunta_0:

    if "knockout" in pregunta_0:
      objs.ind0[1] = "knockout"
      df_fixture_knockout = df1[1].copy()
      df_fixture_quarter = df3[["home", "score", "away"]].iloc[:5].copy()
      df_fixture_semi = df3[["home", "score", "away"]].iloc[5:7].copy()
      df_fixture_third = df3[["home", "score", "away"]].iloc[7].copy()
      df_fixture_final = df3[["home", "score", "away"]].iloc[8].copy()

    if "quarter" in pregunta_0:
      objs.ind0[1] = "quarter"
      df_fixture_knockout = df2.copy()
      df_fixture_quarter = df1[1].copy()
      df_fixture_semi = df3[["home", "score", "away"]].iloc[5:7].copy()
      df_fixture_third = df3[["home", "score", "away"]].iloc[7].copy()
      df_fixture_final = df3[["home", "score", "away"]].iloc[8].copy()

    if "semi" in pregunta_0:
      objs.ind0[1] = "semi"
      df_fixture_knockout = df2.copy()
      df_fixture_quarter = df1[0][["home", "away"]].iloc[:5].copy()
      df_fixture_semi = df1[1].copy()
      df_fixture_third = df3[["home", "score", "away"]].iloc[7].copy()
      df_fixture_final = df3[["home", "score", "away"]].iloc[8].copy()

    if "third" in pregunta_0:
      objs.ind0[1] = "third"
      df_fixture_knockout = df2.copy()
      df_fixture_quarter = df1[0][["home", "away"]].iloc[:5].copy()
      df_fixture_semi = df1[0][["home", "away"]].iloc[5:7].copy()
      df_fixture_third = df1[1].copy()
      df_fixture_final = df3[["home", "score", "away"]].iloc[8].copy()

    if "final" in pregunta_0:
      objs.ind0[1] = "final"
      df_fixture_knockout = df2.copy()
      df_fixture_quarter = df1[0][["home", "away"]].iloc[:5].copy()
      df_fixture_semi = df1[0][["home", "away"]].iloc[5:7].copy()
      df_fixture_third = df1[0][["home", "away"]].iloc[7].copy()
      df_fixture_final = df1[1].copy()

    objs.ind0[0] = "knockout"
    objs.get_winner(df_fixture_knockout, modelo)

    df_fixture_quarter = objs.ordenar_elm_smpl(df_fixture_knockout, df_fixture_quarter)

    objs.ind0[0] = "quarter"
    objs.get_winner(df_fixture_quarter, modelo)

    df_fixture_semi = objs.ordenar_elm_smpl(df_fixture_quarter, df_fixture_semi)

    objs.ind0[0] = "semi"
    objs.get_winner(df_fixture_semi, modelo)

    df_fixture_third = objs.ordenar_elm_smpl(df_fixture_semi, df_fixture_third, tipo="Loser")

    objs.ind0[0] = "third"
    objs.get_winner(df_fixture_third, modelo)

    df_fixture_final = objs.ordenar_elm_smpl(df_fixture_semi, df_fixture_final)

    print(f"\n\n----------\n----------\n\nFINAL - FINAL - FINAL: --- {pregunta_0} ---\n----------\n----------\n")
    objs.ind0[0] = "final"
    objs.get_winner(df_fixture_final, modelo)

    df_ent = pd.concat([df_fixture_knockout[["home", "score_0", "score_1", "away"]], df_fixture_quarter[["home", "score_0", "score_1", "away"]], df_fixture_semi[["home", "score_0", "score_1", "away"]], df_fixture_third[["home", "score_0", "score_1", "away"]], df_fixture_final[["home", "score_0", "score_1", "away"]]], axis=0)

    df_ent.home = df_ent.home.map(objs.id_pais)
    df_ent.away = df_ent.away.map(objs.id_pais)

    return df_ent

  elif "1950" in pregunta_0:

    if "grupo" in pregunta_0:
      objs.ind0[1] = "grupo"
      df_fixture_group_ = agregar_features(df1[1].copy(), dic_agnos, ind=0)
      df_fixture_quarter = df3[["home", "score", "away"]].iloc[:6].copy()

    elif "Fase Final" in pregunta_0:
      objs.ind0[1] = "Fase Final"
      df_fixture_group_ = df2.copy()
      df_fixture_quarter = df1[1].copy()

    objs.ind0[0] = "grupo"
    dict_table = objs.fase_de_grupos(dict_g, df_fixture_group_, modelo)

    df_fixture_quarter = objs.camb_grp_elm_smpl(df_fixture_group_, df_fixture_quarter, dict_table)

    dt = {}
    equipos = pd.concat([df_fixture_quarter.home, df_fixture_quarter.away], axis=0).unique().tolist()
    dt["Fase Final"] = pd.DataFrame(data={"pais": equipos, "Pts": [0, 0, 0, 0]})

    objs.ind0[0] = "Fase Final"
    dict_table = objs.fase_de_grupos(dt, df_fixture_quarter, modelo)

    print(f"\n\n----------\n----------\n\nFINAL - FINAL - FINAL: --- {pregunta_0} ---\n----------\n----------\n")

    paises, puntajes = [], []
    dtable = dict_table["Fase Final"].copy()
    dtable.rename(columns={"Pts.": "Pts"}, inplace=True)
    for linea in dtable.itertuples():
      paises.append(objs.id_pais[linea.pais])
      puntajes.append(linea.Pts)
    df_nombres = pd.DataFrame(data={"pais": paises, "Pts": puntajes}, index=range(len(paises)))
    print(df_nombres)

    df_ent = pd.concat([df_fixture_group_[["home", "score_0", "score_1", "away"]], df_fixture_quarter[["home", "score_0", "score_1", "away"]]], axis=0)

    df_ent.home = df_ent.home.map(objs.id_pais)
    df_ent.away = df_ent.away.map(objs.id_pais)

    return df_ent

  elif "1954" in pregunta_0 or "1958" in pregunta_0 or "1962" in pregunta_0 or "1966" in pregunta_0 or "1970" in pregunta_0:

    if "grupo" in pregunta_0:
      objs.ind0[1] = "grupo"
      df_fixture_group_ = agregar_features(df1[1].copy(), dic_agnos, ind=0)
      df_fixture_quarter = df3[["home", "score", "away"]].iloc[:4].copy()
      df_fixture_semi = df3[["home", "score", "away"]].iloc[4:6].copy()
      df_fixture_third = df3[["home", "score", "away"]].iloc[6].copy()
      df_fixture_final = df3[["home", "score", "away"]].iloc[7].copy()

    elif "quarter" in pregunta_0:
      objs.ind0[1] = "quarter"
      df_fixture_group_ = df2.copy()
      df_fixture_quarter = df1[1].copy()
      df_fixture_semi = df3[["home", "score", "away"]].iloc[4:6].copy()
      df_fixture_third = df3[["home", "score", "away"]].iloc[6].copy()
      df_fixture_final = df3[["home", "score", "away"]].iloc[7].copy()

    elif "semi" in pregunta_0:
      objs.ind0[1] = "semi"
      df_fixture_group_ = df2.copy()
      df_fixture_quarter = df1[0][["home", "away"]].iloc[:4].copy()
      df_fixture_semi = df1[1].copy()
      df_fixture_third = df3[["home", "score", "away"]].iloc[6].copy()
      df_fixture_final = df3[["home", "score", "away"]].iloc[7].copy()

    elif "third" in pregunta_0:
      objs.ind0[1] = "third"
      df_fixture_group_ = df2.copy()
      df_fixture_quarter = df1[0][["home", "away"]].iloc[:4].copy()
      df_fixture_semi = df1[0][["home", "away"]].iloc[4:6].copy()
      df_fixture_third = df1[1].copy()
      df_fixture_final = df3[["home", "score", "away"]].iloc[7].copy()

    elif "final" in pregunta_0:
      objs.ind0[1] = "final"
      df_fixture_group_ = df2.copy()
      df_fixture_quarter = df1[0][["home", "away"]].iloc[0:4].copy()
      df_fixture_semi = df1[0][["home", "away"]].iloc[4:6].copy()
      df_fixture_third = df1[0][["home", "away"]].iloc[6].copy()
      df_fixture_final = df1[1].copy()

    objs.ind0[0] = "grupo"
    dict_table = objs.fase_de_grupos(dict_g, df_fixture_group_, modelo)

    df_fixture_quarter = objs.camb_grp_elm_smpl(df_fixture_group_, df_fixture_quarter, dict_table)

    objs.ind0[0] = "quarter"
    objs.get_winner(df_fixture_quarter, modelo)

    df_fixture_semi = objs.ordenar_elm_smpl(df_fixture_quarter, df_fixture_semi)

    objs.ind0[0] = "semi"
    objs.get_winner(df_fixture_semi, modelo)

    df_fixture_third = objs.ordenar_elm_smpl(df_fixture_semi, df_fixture_third, tipo="Loser")

    objs.ind0[0] = "third"
    objs.get_winner(df_fixture_third, modelo)

    df_fixture_final = objs.ordenar_elm_smpl(df_fixture_semi, df_fixture_final)

    print(f"\n\n----------\n----------\n\nFINAL - FINAL - FINAL: --- {pregunta_0} ---\n----------\n----------\n")
    objs.ind0[0] = "final"
    objs.get_winner(df_fixture_final, modelo)

    df_ent = pd.concat([df_fixture_group_[["home", "score_0", "score_1", "away"]], df_fixture_quarter[["home", "score_0", "score_1", "away"]], df_fixture_semi[["home", "score_0", "score_1", "away"]], df_fixture_third[["home", "score_0", "score_1", "away"]], df_fixture_final[["home", "score_0", "score_1", "away"]]], axis=0)

    df_ent.home = df_ent.home.map(objs.id_pais)
    df_ent.away = df_ent.away.map(objs.id_pais)

    return df_ent
  
  elif "1986" in pregunta_0 or "1990" in pregunta_0 or "1994" in pregunta_0:
    emparejar_equipos_val = emparejar_equipos(objs)

    if "grupo" in pregunta_0:
      objs.ind0[1] = "grupo"
      df_fixture_group_ = agregar_features(df1[1].copy(), dic_agnos, ind=0)
      df_fixture_knockout = knock_out_1986_1994(emparejar_equipos_val, df3).copy()
      df_fixture_quarter = df3[["home", "score", "away"]].iloc[8:12].copy()
      df_fixture_semi = df3[["home", "score", "away"]].iloc[12:14].copy()
      df_fixture_third = df3[["home", "score", "away"]].iloc[14].copy()
      df_fixture_final = df3[["home", "score", "away"]].iloc[15].copy()

    elif "knockout" in pregunta_0:
      objs.ind0[1] = "knockout"
      df_fixture_group_ = df2.copy()
      df_fixture_knockout = knock_out_1986_1994(emparejar_equipos_val, df1[1]).copy()
      df_fixture_quarter = df3[["home", "score", "away"]].iloc[8:12].copy()
      df_fixture_semi = df3[["home", "score", "away"]].iloc[12:14].copy()
      df_fixture_third = df3[["home", "score", "away"]].iloc[14].copy()
      df_fixture_final = df3[["home", "score", "away"]].iloc[15].copy()

    elif "quarter" in pregunta_0:
      objs.ind0[1] = "quarter"
      df_fixture_group_ = df2.copy()
      df_fixture_knockout = knock_out_1986_1994(emparejar_equipos_val, df1[0]).copy()
      df_fixture_quarter = df1[1].copy()
      df_fixture_semi = df3[["home", "score", "away"]].iloc[12:14].copy()
      df_fixture_third = df3[["home", "score", "away"]].iloc[14].copy()
      df_fixture_final = df3[["home", "score", "away"]].iloc[15].copy()

    elif "semi" in pregunta_0:
      objs.ind0[1] = "semi"
      df_fixture_group_ = df2.copy()
      df_fixture_knockout = knock_out_1986_1994(emparejar_equipos_val, df1[0]).copy()
      df_fixture_quarter = df1[0][["home", "away"]].iloc[8:12].copy()
      df_fixture_semi = df1[1].copy()
      df_fixture_third = df3[["home", "score", "away"]].iloc[14].copy()
      df_fixture_final = df3[["home", "score", "away"]].iloc[15].copy()

    elif "third" in pregunta_0:
      objs.ind0[1] = "third"
      df_fixture_group_ = df2.copy()
      df_fixture_knockout = knock_out_1986_1994(emparejar_equipos_val, df1[0]).copy()
      df_fixture_quarter = df1[0][["home", "away"]].iloc[8:12].copy()
      df_fixture_semi = df1[0][["home", "away"]].iloc[12:14].copy()
      df_fixture_third = df1[1].copy()
      df_fixture_final = df3[["home", "score", "away"]].iloc[15].copy()

    elif "final" in pregunta_0:
      objs.ind0[1] = "final"
      df_fixture_group_ = df2.copy()
      df_fixture_knockout = knock_out_1986_1994(emparejar_equipos_val, df1[0]).copy()
      df_fixture_quarter = df1[0][["home", "away"]].iloc[8:12].copy()
      df_fixture_semi = df1[0][["home", "away"]].iloc[12:14].copy()
      df_fixture_third = df1[0][["home", "away"]].iloc[14].copy()
      df_fixture_final = df1[1].copy()

    objs.ind0[0] = "grupo"
    dict_table = objs.fase_de_grupos(dict_g, df_fixture_group_, modelo)

    df_fixture_knockout = objs.camb_grp_elm_smpl(df_fixture_group_, df_fixture_knockout, dict_table)

    objs.ind0[0] = "knockout"
    objs.get_winner(df_fixture_knockout, modelo)

    df_fixture_quarter = objs.ordenar_elm_smpl(df_fixture_knockout, df_fixture_quarter)

    objs.ind0[0] = "quarter"
    objs.get_winner(df_fixture_quarter, modelo)

    df_fixture_semi = objs.ordenar_elm_smpl(df_fixture_quarter, df_fixture_semi)

    objs.ind0[0] = "semi"
    objs.get_winner(df_fixture_semi, modelo)

    df_fixture_third = objs.ordenar_elm_smpl(df_fixture_semi, df_fixture_third, tipo="Loser")

    objs.ind0[0] = "third"
    objs.get_winner(df_fixture_third, modelo)

    df_fixture_final = objs.ordenar_elm_smpl(df_fixture_semi, df_fixture_final)

    print(f"\n\n----------\n----------\n\nFINAL - FINAL - FINAL: --- {pregunta_0} ---\n----------\n----------\n")
    objs.ind0[0] = "final"
    objs.get_winner(df_fixture_final, modelo)

    df_ent = pd.concat([df_fixture_group_[["home", "score_0", "score_1", "away"]], df_fixture_knockout[["home", "score_0", "score_1", "away"]], df_fixture_quarter[["home", "score_0", "score_1", "away"]], df_fixture_semi[["home", "score_0", "score_1", "away"]], df_fixture_third[["home", "score_0", "score_1", "away"]], df_fixture_final[["home", "score_0", "score_1", "away"]]], axis=0)

    df_ent.home = df_ent.home.map(objs.id_pais)
    df_ent.away = df_ent.away.map(objs.id_pais)

    return df_ent
    
  elif "1974" in pregunta_0 or "1978" in pregunta_0:

    if "grupo" in pregunta_0:
      objs.ind0[1] = "grupo"
      df_fixture_group_ = agregar_features(df1[1].copy(), dic_agnos, ind=0)
      df_fixture_doce = df3[["home", "score", "away"]].iloc[:12].copy()
      df_fixture_third = df3[["home", "score", "away"]].iloc[12].copy()
      df_fixture_final = df3[["home", "score", "away"]].iloc[13].copy()

    elif "doce" in pregunta_0:
      objs.ind0[1] = "doce"
      df_fixture_group_ = df2.copy()
      df_fixture_doce = df1[1].copy()
      df_fixture_third = df3[["home", "score", "away"]].iloc[12].copy()
      df_fixture_final = df3[["home", "score", "away"]].iloc[13].copy()

    elif "third" in pregunta_0:
      objs.ind0[1] = "third"
      df_fixture_group_ = df2.copy()
      df_fixture_doce = df1[0][["home", "away"]].iloc[:12].copy()
      df_fixture_third = df1[1].copy()
      df_fixture_final = df3[["home", "score", "away"]].iloc[13].copy()

    elif "final" in pregunta_0:
      objs.ind0[1] = "final"
      df_fixture_group_ = df2.copy()
      df_fixture_doce = df1[0][["home", "away"]].iloc[:12].copy()
      df_fixture_third = df1[0][["home", "away"]].iloc[12].copy()
      df_fixture_final = df1[1].copy()

    dict_g_1974G0 = {"Grupo 1": dict_g.get("Grupo 1"),
                     "Grupo 2": dict_g.get("Grupo 2"),
                     "Grupo 3": dict_g.get("Grupo 3"),
                     "Grupo 4": dict_g.get("Grupo 4")}

    objs.ind0[0] = "grupo"
    dict_table = objs.fase_de_grupos(dict_g_1974G0, df_fixture_group_, modelo)

    objs.n_partidos = df_fixture_group_.shape[0] + df_fixture_doce.shape[0] + 1
    df_fixture_doce = objs.camb_grp_elm_smpl(df_fixture_group_, df_fixture_doce, dict_table, year=int(pregunta_0[-4:]))

    objs.ind0[0] = "doce"
    dict_table = objs.fase_de_grupos(df_fixture_doce[1], df_fixture_doce[0].iloc[:, :-3], modelo)

    df_fixture_third = objs.camb_grp_elm_smpl(df_fixture_doce[0].iloc[:, :-3], df_fixture_third, dict_table)

    objs.ind0[0] = "third"
    if df_fixture_third.shape[0] > 1:
      objs.get_winner(df_fixture_third.iloc[0:-1], modelo)
    elif df_fixture_third.shape[0] == 1:
      objs.get_winner(df_fixture_third, modelo)

    df_fixture_final = objs.camb_grp_elm_smpl(df_fixture_doce[0].iloc[:, :-3], df_fixture_final, dict_table)

    print(f"\n\n----------\n----------\n\nFINAL - FINAL - FINAL: --- {pregunta_0} ---\n----------\n----------\n")
    objs.ind0[0] = "final"
    objs.get_winner(df_fixture_final, modelo)

    df_ent = pd.concat([df_fixture_group_[["home", "score_0", "score_1", "away"]], df_fixture_doce[0][["home", "score_0", "score_1", "away"]], df_fixture_third[["home", "score_0", "score_1", "away"]], df_fixture_final[["home", "score_0", "score_1", "away"]]], axis=0)

    df_ent.home = df_ent.home.map(objs.id_pais)
    df_ent.away = df_ent.away.map(objs.id_pais)

    return df_ent

  elif "1982" in pregunta_0:

    if "grupo" in pregunta_0:
      objs.ind0[1] = "grupo"
      df_fixture_group_ = agregar_features(df1[1].copy(), dic_agnos, ind=0)
      df_fixture_doce = df3[["home", "score", "away"]].iloc[:12].copy()
      df_fixture_semi = df3[["home", "score", "away"]].iloc[12:14].copy()
      df_fixture_third = df3[["home", "score", "away"]].iloc[14].copy()
      df_fixture_final = df3[["home", "score", "away"]].iloc[15].copy()

    elif "doce" in pregunta_0:
      objs.ind0[1] = "doce"
      df_fixture_group_ = df2.copy()
      df_fixture_doce = df1[1].copy()
      df_fixture_semi = df3[["home", "score", "away"]].iloc[12:14].copy()
      df_fixture_third = df3[["home", "score", "away"]].iloc[14].copy()
      df_fixture_final = df3[["home", "score", "away"]].iloc[15].copy()

    elif "semi" in pregunta_0:
      objs.ind0[1] = "semi"
      df_fixture_group_ = df2.copy()
      df_fixture_doce = df1[0][["home", "away"]].iloc[:12].copy().copy()
      df_fixture_semi = df1[1].copy()
      df_fixture_third = df3[["home", "score", "away"]].iloc[14].copy()
      df_fixture_final = df3[["home", "score", "away"]].iloc[15].copy()

    elif "third" in pregunta_0:
      objs.ind0[1] = "third"
      df_fixture_group_ = df2.copy()
      df_fixture_doce = df1[0][["home", "away"]].iloc[:12].copy()
      df_fixture_semi = df1[0][["home", "away"]].iloc[12:14].copy()
      df_fixture_third = df1[1].copy()
      df_fixture_final = df3[["home", "score", "away"]].iloc[15].copy()

    elif "final" in pregunta_0:
      objs.ind0[1] = "final"
      df_fixture_group_ = df2.copy()
      df_fixture_doce = df1[0][["home", "away"]].iloc[:12].copy()
      df_fixture_semi = df1[0][["home", "away"]].iloc[12:14].copy()
      df_fixture_third = df1[0][["home", "away"]].iloc[14].copy()
      df_fixture_final = df1[1].copy()

    dict_g_1982G0 = {"Grupo 1": dict_g.get("Grupo 1"),
                     "Grupo 2": dict_g.get("Grupo 2"),
                     "Grupo 3": dict_g.get("Grupo 3"),
                     "Grupo 4": dict_g.get("Grupo 4"),
                     "Grupo 5": dict_g.get("Grupo 5"),
                     "Grupo 6": dict_g.get("Grupo 6")}

    objs.ind0[0] = "grupo"
    dict_table = objs.fase_de_grupos(dict_g_1982G0, df_fixture_group_, modelo)

    df_fixture_doce = objs.camb_grp_elm_smpl(df_fixture_group_, df_fixture_doce, dict_table, year=int(pregunta_0[-4:]))

    objs.ind0[0] = "doce"
    dict_table = objs.fase_de_grupos(df_fixture_doce[1], df_fixture_doce[0].iloc[:, :-3], modelo)

    objs.n_partidos = df_fixture_group_.shape[0] + df_fixture_doce[0].shape[0] + 1
    
    df_fixture_semi = objs.camb_grp_elm_smpl(df_fixture_doce[0].iloc[:, :-3], df_fixture_semi, dict_table)

    objs.ind0[0] = "semi"
    objs.get_winner(df_fixture_semi, modelo)

    df_fixture_third = objs.ordenar_elm_smpl(df_fixture_semi, df_fixture_third, tipo="Loser")

    objs.ind0[0] = "third"
    objs.get_winner(df_fixture_third, modelo)

    df_fixture_final = objs.ordenar_elm_smpl(df_fixture_semi, df_fixture_final)

    print(f"\n\n----------\n----------\n\nFINAL - FINAL - FINAL: --- {pregunta_0} ---\n----------\n----------\n")
    objs.ind0[0] = "final"
    objs.get_winner(df_fixture_final, modelo)

    df_ent = pd.concat([df_fixture_group_[["home", "score_0", "score_1", "away"]], df_fixture_doce[0][["home", "score_0", "score_1", "away"]], df_fixture_semi[["home", "score_0", "score_1", "away"]], df_fixture_third[["home", "score_0", "score_1", "away"]], df_fixture_final[["home", "score_0", "score_1", "away"]]], axis=0)

    df_ent.home = df_ent.home.map(objs.id_pais)
    df_ent.away = df_ent.away.map(objs.id_pais)

    return df_ent
