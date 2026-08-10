# ["PJ", "GR", "GA", "VI", "PG", "PTS"]

# PJ: Partidos Jugados
# GR: Goles Recibidos
# GA: Goles Anotados
# VI: Vallas Invictas
# PG: Partidos con Goles Anotados
# PTS: Puntos Obtenidos

# SOT
# PKATT
# PKATTALLOW

# country

# 0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-

import pandas as pd

# Opción 1: Usando funciones definidas (más clara)
def calc_gkps1(group):
    VI = group['VI'].sum()
    PJ = group['PJ'].sum()
    GR = group['GR'].sum()
    PKATTALLOW = group['PKATTALLOW'].sum() if 'PKATTALLOW' in group.columns else 0
    return ((VI/PJ)*60) + (40 - ((GR - PKATTALLOW)/PJ)*10) if PJ > 0 else 0

def calc_mds1(group):
    VI = group['VI'].sum()
    PJ = group['PJ'].sum()
    GR = group['GR'].sum()
    PKATTALLOW = group['PKATTALLOW'].sum() if 'PKATTALLOW' in group.columns else 0
    return ((VI/PJ)*50) + (50 - (((GR + PKATTALLOW)/PJ)*10)) if PJ > 0 else 0

def calc_mos1(group):
    GA = group['GA'].sum()
    PJ = group['PJ'].sum()
    SOT = group['SOT'].sum() if 'SOT' in group.columns else 0
    PKATT = group['PKATT'].sum() if 'PKATT' in group.columns else 0
    return ((GA/PJ)*0.5 + (SOT/PJ)*0.3 + ((GA - PKATT)/SOT)*0.2) if PJ > 0 and SOT > 0 else 0

def calc_mms1(group):
    PTS = group['PTS'].sum()
    PJ = group['PJ'].sum()
    PG = group['PG'].sum()
    SOT = group['SOT'].sum() if 'SOT' in group.columns else 0
    return (((PTS/(PJ*3))*40) + ((PG/PJ)*40) + ((SOT/PJ)*2)) if PJ > 0 else 0

# Aplicar las funciones
dfEspecializado = dfBase.groupby('country').apply(
    lambda g: pd.Series({
        'gkps1': calc_gkps1(g),
        'mds1': calc_mds1(g),
        'mos1': calc_mos1(g),
        'mms1': calc_mms1(g)
    })
).reset_index()

# Unir con el DataFrame original
dfBase = dfBase.merge(dfEspecializado, on='country', how='left')
