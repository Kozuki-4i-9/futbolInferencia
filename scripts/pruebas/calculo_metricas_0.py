# ["PJ", "GR", "GA", "VI", "PG", "PTS"]

# PJ: Partidos Jugados
# GR: Goles Recibidos
# GA: Goles Anotados
# PG: Partidos con Goles Anotados
# PTS: Puntos Obtenidos

# VI: Vallas Invictas Número de partidos en los que el equipo NO recibió goles
# SOT: Número de disparos que TU equipo hace y van entre los tres palos (a portería rival)
# PKATT: Número de penaltis que TU equipo ha lanzado (o recibido a favor) en toda la temporada
# PKATTALLOW: Número de penaltis que TU equipo ha concedido al rival (penaltis en su propia contra)

# (VI, SOT, PKATT, PKATTALLOW)

# country





import pandas as pd
import numpy as np

OPTIONAL_METRICS = None

def formar_dataset_real():
   pass

def calc_gkps(PJ_i, GC_i, VI, PKATTALLOW): # PEND (falta docstring)
    VI = VI.sum()
    PJ = PJ_i.sum()
    GR = GC_i.sum()
    PKATTALLOW = PKATTALLOW.sum()
    return ((VI/PJ)*60) + (40 - ((GR - PKATTALLOW)/PJ)*10) if PJ > 0 else 0

def calc_mds(PJ_i, GC_i, VI, PKATTALLOW): # PEND (falta docstring)
    VI = VI.sum()
    PJ = PJ_i.sum()
    GR = GC_i.sum()
    PKATTALLOW = PKATTALLOW.sum()
    return ((VI/PJ)*50) + (50 - (((GR + PKATTALLOW)/PJ)*10)) if PJ > 0 else 0

def calc_mos(PJ_i, GF_i, SOT, PKATT): # PEND (falta docstring)
    GA = GF_i.sum()
    PJ = PJ_i.sum()
    SOT = SOT.sum()
    PKATT = PKATT.sum()
    return ((GA/PJ)*0.5 + (SOT/PJ)*0.3 + ((GA - PKATT)/SOT)*0.2) if PJ > 0 and SOT > 0 else 0

def calc_mms(PJ_i, PTS_i, PG_i, SOT): # PEND (falta docstring)
    PTS = PJ_i.sum()
    PJ = PTS_i.sum()
    PG = PG_i.sum()
    SOT = SOT.sum()
    return (((PTS/(PJ*3))*40) + ((PG/PJ)*40) + ((SOT/PJ)*2)) if PJ > 0 else 0

def calc_rate(df, tipo, num=0): # PEND (falta docstring)
  busqueda = tipo + "_" + str(num)
  if(busqueda in OPTIONAL_METRICS):
    return (df[busqueda] / df[f"PJ_{num}"].replace(0, 0.085)) * 10
  else:
    return np.nan

def obtener_diferencia(df, num=0): # PEND (falta docstring)
  if(('GC' + '_' + f'{num}' in OPTIONAL_METRICS) and ('GF' + '_' + f'{num}' in OPTIONAL_METRICS)):
    return df[f'GF_{num}'] - df[f'GC_{num}']
  else:
    return np.nan

def calculo_metricas_0(df_desempegno, agno=None): # PEND (falta docstring)
    global opcionales
    columnas = []

    if((OPTIONAL_METRICS!=[]) and opcionales is None):
        opcionales = formar_dataset_real("proxy_desempegno", agno)

    paises_unicos = pd.concat([df_desempegno['home'], df_desempegno['away']]).unique()

    insumos_apuntados = opcionales[opcionales['pais'].isin(paises_unicos)]

    df1 = df_desempegno.copy()

    idxaway = df1.columns.tolist().index('away')
    homes = df1.iloc[:,:idxaway-1] # PEND se trata de evitar tomar score_0 mediante este -1
    aways = df1.iloc[:,idxaway:-1] # PEND se trata de evitar tomar score_1 mediante este -1

    homes.rename(columns={'home':'pais'},inplace=True)
    aways.rename(columns={'away':'pais'},inplace=True)

    apuntar_a_home = homes.merge(insumos_apuntados,on="pais",how="inner")
    apuntar_a_away = aways.merge(insumos_apuntados,on="pais",how="inner")

    df_nuevo = pd.DataFrame()
    equipos = {0:'home', 1:'away'}

    for idx, trab in enumerate([apuntar_a_home, apuntar_a_away]):

        dfEspecializado = trab.groupby('pais', as_index=False).apply(
            lambda g: pd.Series({
                f'gkps_{idx}': calc_gkps(g[f'PJ_{idx}'], g[f'GC_{idx}'], g['VI'], g['PKATTALLOW']),
                f'mds_{idx}': calc_mds(g[f'PJ_{idx}'], g[f'GC_{idx}'], g['VI'], g['PKATTALLOW']),
                f'mos_{idx}': calc_mos(g[f'PJ_{idx}'], g[f'GF_{idx}'], g['SOT'], g['PKATT']),
                f'mms_{idx}': calc_mms(g[f'PJ_{idx}'], g[f'PTS_{idx}'], g[f'PG_{idx}'], g['SOT']),
                f'rate_GC_{idx}': calc_rate(g[f'GC_{idx}'], 'GC', num=idx),
                f'rate_GF_{idx}': calc_rate(g[f'GF_{idx}'], 'GF', num=idx),
                f'D_{idx}': obtener_diferencia(g, num=idx),
            })
        ).reset_index(drop=True)

        trab.drop(columns=['VI', 'SOT', 'PKATT', 'PKATTALLOW'], inplace=True)

        df_provisional = trab.merge(dfEspecializado, on='pais', how='inner')
        df_provisional.rename(columns={'pais':equipos[idx]},inplace=True)

        nuevas_columnas = [f'gkps_{idx}', f'mds_{idx}', f'mos_{idx}', f'mms_{idx}',
                          f'rate_GC_{idx}', f'rate_GF_{idx}', f'D_{idx}']

        df_nuevo = pd.concat([df_nuevo, df_provisional], axis=1, ignore_index=True)
        columnas = [*columnas, *df_provisional.columns]

    df_nuevo.columns = columnas
    return df_nuevo