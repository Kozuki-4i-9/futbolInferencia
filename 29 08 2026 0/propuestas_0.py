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

def calc_gkps(group):
    VI = group['VI'].sum()
    PJ = group['PJ'].sum()
    GR = group['GR'].sum()
    PKATTALLOW = group['PKATTALLOW'].sum() if 'PKATTALLOW' in group.columns else 0
    return ((VI/PJ)*60) + (40 - ((GR - PKATTALLOW)/PJ)*10) if PJ > 0 else 0

def calc_mds(group):
    VI = group['VI'].sum()
    PJ = group['PJ'].sum()
    GR = group['GR'].sum()
    PKATTALLOW = group['PKATTALLOW'].sum() if 'PKATTALLOW' in group.columns else 0
    return ((VI/PJ)*50) + (50 - (((GR + PKATTALLOW)/PJ)*10)) if PJ > 0 else 0

def calc_mos(group):
    GA = group['GA'].sum()
    PJ = group['PJ'].sum()
    SOT = group['SOT'].sum() if 'SOT' in group.columns else 0
    PKATT = group['PKATT'].sum() if 'PKATT' in group.columns else 0
    return ((GA/PJ)*0.5 + (SOT/PJ)*0.3 + ((GA - PKATT)/SOT)*0.2) if PJ > 0 and SOT > 0 else 0

def calc_mms(group):
    PTS = group['PTS'].sum()
    PJ = group['PJ'].sum()
    PG = group['PG'].sum()
    SOT = group['SOT'].sum() if 'SOT' in group.columns else 0
    return (((PTS/(PJ*3))*40) + ((PG/PJ)*40) + ((SOT/PJ)*2)) if PJ > 0 else 0

dfEspecializado = dfBase.groupby('country').apply(
    lambda g: pd.Series({
        'gkps1': calc_gkps1(g),
        'mds1': calc_mds1(g),
        'mos1': calc_mos1(g),
        'mms1': calc_mms1(g)
    })
).reset_index()

dfBase = dfBase.merge(dfEspecializado, on='country', how='left')







el modelo se entreno con los datos de los partidos de todos los mundiales (947 row hasta 2022) con esta estructura de inputs:

home   PTS_0   PG_0   PP_0   PE_0   D_0   score_0   ts_GF_0   ts_GC_0   away   PTS_1   PG_1   PP_1   PE_1   D_1   score_1   ts_GF_1   ts_GC_1
x, y = calculo_metricas_0(df1), df1[["score_0", "score_1"]] 

se le aplica una capa Embedding de tensorflow
x0 = objs.hacer_embedding_a_equipos(data_0) # home   -   x1 = objs.hacer_embedding_a_equipos(data_1) # away

se le aplica una funcion para tomar lotes de entreda tamaño (n) y la correspondiente salida (modelo timestep para LSTMs)

x_tr_h, x_tr_a = objs.c_s(x0[:607]), objs.c_s(x1[:607])
xtr_home, xtr_away = x_tr_h[0], x_tr_a[0]
ytr_home, ytr_away = x_tr_h[1], x_tr_a[1]
X_tr = np.concatenate((xtr_home, xtr_away), axis=0)
Y_tr = np.concatenate((ytr_home, ytr_away), axis=0)
X_tr.shape, Y_tr.shape   ->  (1204, 5, 10) (1204,)

el modelo usado hasta ahora, probado y aprobado el diciembre de 2024 es:

model = tf.keras.Sequential([
    tf.keras.layers.LSTM(100, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.Dropout(0.20),
    tf.keras.layers.LSTM(57, return_sequences=True),
    tf.keras.layers.Dropout(0.40),
    tf.keras.layers.LSTM(25, return_sequences=False),
    tf.keras.layers.Dropout(0.15),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)  # Salida para regresión
])
# Compilar el modelo
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mae'])

model.fit(X_tr, Y_tr,  batch_size=16, validation_data=(X_val, Y_val), epochs=110, callbacks=[TensorMetricas]) 





# pasos a para agergar metriicas de rendimiento complejas:

0- usar selenium para bajar datos extra (goal_keeper_score, mean_defense_score, mean_ofense_score, mean_midfield_score) de 
   (https://fbref.com/en/comps/1/1982/keepers/1982-World-Cup-Stats) generando un df de salida
1- unir ese df (paso anterior) con el de (C:\Users\USUARIO\Trabajo\proyectos\mundiales\futbolInferencia\data\partidos\partidos.csv) y 
   (C:\Users\USUARIO\Trabajo\proyectos\mundiales\futbolInferencia\data\grupos\grupos_mundiales)
2- recrear y adecuar el (modulo_9.py) para entrenar un modelo con la union del paso anterior (adecuar funcion (calculo_metricas_0) para calculo 
   de metricas de rendiimiento complejas (goal_keeper_score, mean_defense_score, mean_ofense_score, mean_midfield_score))
3- adecuar (modulo_9.py) para entregar siempre al modelo en (intsc_prob_goles) (linea 1169 de (modulo_9.py)) una data de dimension (None,946,13)
  (home   PTS_0   PG_0   PP_0   PE_0   D_0   score_0   ts_GF_0   ts_GC_0   gkps_0   mds_0   mos_0   mms_0)
  (away   PTS_1   PG_1   PP_1   PE_1   D_1   score_1   ts_GF_1   ts_GC_1   gkps_1   mds_1   mos_1   mms_1)
4- dejar el modelo del paso 3- en la direccion de la que (modelo_9.py) lo toma en la linea 1894 1895

# 11 08 2026

0- nada de (formar_dataset_real) llega con parametros de rendimiento extra
1- los parametros de rendimiento extra se agregan solo a df2 (fase de grupos del fixtura del mundial jugado ); df1 (partidos reales del mundial jugado (ambas fases))
2- los parametros de rendimiento extra se agregan con agregar_features, que depende de (dic_agnos) que depende de partidos de eliminatorias (formar_dataset_real("clasificaciones"))
3- con (seleccionar) se lleba la estructura de metricas de rendimiento base de clasificaciones al resto de las fases 
4- con (calculo_metricas_0) actualmente se quitan (PJ_0   GF_0   GC_0   D_0) y se agrega (ts_GF_0   ts_GC_0) a (home  PTS_0  PG_0  PP_0  PE_0  D_0  score_0) formando 9 
   metricas para entrenamiento
   NOTA: (calculo_metricas_0) solo se usa para la prediccion con (self.intsc_prob_goles), el df que pasa a la siguiente fase es (df_fixture_) que entra a 
   (calculo_metricas_0) al inicio de la fase
5- con (calculo_metricas_0) deberian agregarse n columnas como un lienzo fijo para metricas extra y rellenar de forma opcional las que correspondan a las escogidas por 
   los usuarios sustituyendo lineas como ("ts_GF_0": ts_home_GF, "ts_GC_0": ts_home_GC, "ts_GF_1": ts_away_GF, "ts_GC_1": ts_away_GC,) con diccionarios dinamico (**{}) 
   para metricas opcionales en lienzo fijo

# 29 08 2026 0

- (camb_grp_elm_smpl) recibe los siguientes argumentos (df_fixture_group_, df_fixture_knockout, dict_table) siendo (dict_table) el diccionario extraido de 
  (grupos_anio_interes) como diccionario con clave ("Grupo X") y valor df con metricas de rendimiento base seteadas a 0. El uso que le da es busqueda y posicionamiento
  de equipos segun bloquees del fixture y 
- (data_table) no se unsa en ninguna funcion despues de (camb_grp_elm_smpl)

- definir dependencias entradas y salidas de todas las funciones de torneo
- actualizar función orm para tomar nueva tabla métricas proxy
- crear scraper selenium para métricas proxy