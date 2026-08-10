import pandas as pd

df0 = pd.read_csv('directorio0/directorio1/archivo.csv')

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

for filna in df0.itertuples():
	# goal keeper score
	gkps0 = 10 - (3 * (filna.GR/filna.PJ)) + (filna.VI/filna.PJ) # goalkeeper_score
	gkps1 = ((filna.VI/filna.PJ)*60) + (40 - ((filna.GR - filna.PKATTALLOW)/filna.PJ)*10)

	# mean defense score
	mds0 = 10 - (3*(filna.GR/filna.PJ)) + (filna.VI/filna.PJ)
	mds1 = ((filna.VI/filna.PJ)*50) + (50-(((filna.GR + filna.PKATTALLOW)/filna.PJ)*10))

	# mean ofense score
	mos0 = (2.5*(filna.GA/filna.PJ)) + (filna.PG/filna.PJ)
	mos1 = (filna.GA/filna.PJ)*0.5 + (filna.SOT/filna.PJ)*0.3 + ((filna.GA - filna.PKATT)/filna.SOT)*0.2

	# mean midfield score
	mms0 = 5 + ((filna.GA/filna.PJ) - (filna.GR/filna.PJ)) + (3*(filna.PTS/(filna.PJ*3)))
	mms1 = ((filna.PTS/(filna.PJ*3))*40) + ((filna.PG/filna.PJ)*40) + ((filna.SOT/filna.PJ)*2)

	df1[df1["country"]==country][]

# -1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-

dfEspecilizado = dfBase.groupby('country').agg(
gkps1 = lambda filna: ((filna.VI/filna.PJ)*60) + (40 - ((filna.GR - filna.PKATTALLOW)/filna.PJ)*10),
mds1 = lambda filna: ((filna.VI/filna.PJ)*50) + (50-(((filna.GR + filna.PKATTALLOW)/filna.PJ)*10)),
mos1 = lambda filna: (filna.GA/filna.PJ)*0.5 + (filna.SOT/filna.PJ)*0.3 + ((filna.GA - filna.PKATT)/filna.SOT)*0.2,
mms1 = lambda filna: ((filna.PTS/(filna.PJ*3))*40) + ((filna.PG/filna.PJ)*40) + ((filna.SOT/filna.PJ)*2)
	).reset_index()

dfBase.mergeg(dfEspecilizado, on='country', how='left')

# -2-2-2-2-2-2-2-2-2-2-2-2-2-2-2-2-2-2-2-2-2-2-2-2-2-2-2-2-2-2-2-2-2-2-2-2-2-2-2-2-2-2-2-2-2-2-2-2-2-2-

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