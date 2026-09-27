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

# 0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-

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
    homes = df1.iloc[:,:idxaway-1]
    aways = df1.iloc[:,idxaway:-1]

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
        df_provisional.rename(columns={'pais':equipos[idx]},inplace=True);

        nuevas_columnas = [f'gkps_{idx}', f'mds_{idx}', f'mos_{idx}', f'mms_{idx}',
                          f'rate_GC_{idx}', f'rate_GF_{idx}', f'D_{idx}']

        df_nuevo = pd.concat([df_nuevo, df_provisional], axis=1, ignore_index=True)
        columnas = [*columnas, *df_provisional.columns]

    df_nuevo.columns = columnas
    return df_nuevo












# directrices de entrenamiento basado en el archivo (modulo_i.py)

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
Y_tr = np.concatenate((ytr_home, ytr_away), axis=0) # PEND probar si podemos hacer shuffle
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
# 0-)  usar selenium para bajar datos extra (goal_keeper_score, mean_defense_score, mean_ofense_score, mean_midfield_score) de 
#      (https://fbref.com/en/comps/1/1982/keepers/1982-World-Cup-Stats) generando un df de salida - solo se buscaran (VI, SOT, PKATT, PKATTALLOW)
# 1-)  unir ese df (paso anterior) con el de (C:\Users\USUARIO\Trabajo\proyectos\mundiales\futbolInferencia\data\partidos\partidos.csv) y 
#      (C:\Users\USUARIO\Trabajo\proyectos\mundiales\futbolInferencia\data\grupos\grupos_mundiales) - no se va a unir nada se creara una nueva 
#      tabla con migrate y lo que ya se tiene en models.py
# 2-)  recrear y adecuar el (modulo_9.py) para entrenar un modelo con la union del paso anterior (adecuar funcion (calculo_metricas_0) para calculo 
#      de metricas de rendiimiento complejas (goal_keeper_score, mean_defense_score, mean_ofense_score, mean_midfield_score))
# 3-)  adecuar (modulo_9.py) para entregar siempre al modelo en (intsc_prob_goles) (linea 1169 de (modulo_9.py)) una data de dimension (None,946,13)
#      (home   PTS_0   PG_0   PP_0   PE_0   D_0   score_0   ts_GF_0   ts_GC_0   gkps_0   mds_0   mos_0   mms_0)
#      (away   PTS_1   PG_1   PP_1   PE_1   D_1   score_1   ts_GF_1   ts_GC_1   gkps_1   mds_1   mos_1   mms_1)
# 4-)   dejar el modelo del paso 3- en la direccion de la que (modelo_9.py) lo toma en la linea 1894 1895





# 11 08 2026
# 0-)  nada de (formar_dataset_real) llega con parametros de rendimiento extra - ahora si ya se tiene una (elif ind0 == "proxy_desempegno":)
# 1-)  los parametros de rendimiento extra se agregan solo a df2 (fase de grupos del fixtura del mundial jugado ); df1 (partidos reales del mundial jugado (ambas fases))
# 2-)  los parametros de rendimiento extra se agregan con agregar_features, que depende de (dic_agnos) que depende de partidos de eliminatorias (formar_dataset_real("clasificaciones"))
# 3-)  con (seleccionar) se lleba la estructura de metricas de rendimiento base de clasificaciones al resto de las fases 
# 4-)  con (calculo_metricas_0) actualmente se quitan (PJ_0   GF_0   GC_0   D_0) y se agrega (ts_GF_0   ts_GC_0) a (home  PTS_0  PG_0  PP_0  PE_0  D_0  score_0) formando 9 
#      metricas para entrenamiento
#      NOTA: (calculo_metricas_0) solo se usa para la prediccion con (self.intsc_prob_goles), el df que pasa a la siguiente fase es (df_fixture_) que entra a 
#      (calculo_metricas_0) al inicio de la fase - sin ser modificado
# 5-)  con (calculo_metricas_0) deberian agregarse n columnas como un lienzo fijo para metricas extra y rellenar de forma opcional las que correspondan a las escogidas por 
#      los usuarios sustituyendo lineas como ("ts_GF_0": ts_home_GF, "ts_GC_0": ts_home_GC, "ts_GF_1": ts_away_GF, "ts_GC_1": ts_away_GC,) con diccionarios dinamico (**{}) 
#      para metricas opcionales en lienzo fijo - ya se ha actualizado (calculo_metricas_0) para esto y falta incluir el drop de metricas base tambien en lienzo fijo





# 29 08 2026 0
# 0-)  (camb_grp_elm_smpl) recibe los siguientes argumentos (df_fixture_group_, df_fixture_knockout, dict_table) siendo (dict_table) el diccionario extraido de 
#      (grupos_anio_interes) como diccionario con clave ("Grupo X") y valor df con metricas de rendimiento base seteadas a 0. El uso que le da es busqueda y posicionamiento
#      de equipos segun bloquees del fixture y - lo que sale de (grupos_anio_interes) no afecta a metricas fase a fase de df2 o df3 que usa (seleccionar) solo se emplean en (fase_de_grupos) en sus campos (pais,Pts) 
#      para definir el lugar de cada equipo segun su desempeño
# 1-)  (data_table) no se unsa en ninguna funcion despues de (camb_grp_elm_smpl)
# 2-)  definir dependencias entradas y salidas de todas las funciones de torneo - ya esta
# 3-)  actualizar función orm para tomar nueva tabla métricas proxy - ya esta
# 4-)  crear scraper selenium para métricas proxy





# 30 08 2026 0 (PROMPTS)
# deben scrapearse solo estos parametros para (VI, SOT, PKATT, PKATTALLOW) para fases de clasificacion o en su defecto del mundial a predecir

# dame un resumen en que:
#     - me definas de nuevo: VI, SOT, PKATT, PKATTALLOW
#     - me definas la interpretacion oponent stats de estas variables en la tabla (intuyo que es: Argelia->sota (tiros a puerta recibidos por los rivales de argelia), pero tu aclaramelo)
#     - como atenderias el caso de que, tengo un df pandas con solo esas 4 variables VI, SOT, PKATT, PKATTALLOW por equipo y, tras una serie de partidos donde cambian score_home (goles de home) y score_away (gooles de away) en otro df (donde que tambien cuento con estas metricas "PTS_0", "PJ_0", "PG_0", "PP_0", "PE_0", "GF_0", "GC_0", "D_0" para home y "PTS_1", "PJ_1", "PG_1", "PP_1", "PE_1", "GF_1", "GC_1", "D_1" para away), como puedo ajustar matematicamente y a nivel de pandas, las variables del 1er df VI, SOT, PKATT, PKATTALLOW 





# 03 09 2026 0
# https://chat.deepseek.com/share/vthng5glruvh2jzv0z
# https://chat.deepseek.com/share/jyv6jrfjj9lxfpoi6n





# 04 09 2026 0 (PROMPTS)
# muestrame aqui una version de la funcion (funcion_tabla_desempegno) del archivo (modulo_11.py) y actualizala agregandole la logica del script de (C:\Users\USUARIO\Trabajo\proyectos\mundiales\futbolInferencia\03 09 2026 0\scripts\imputar_opcionales.py). NO ACTUALICES NADA EN EL ARCHIVO ORIGINAL SOLO MUESTRAME LA VERSION QUE PIDO Y SI SE REQUIERE ACTUALIZAR ALGO EN EL SCRIPT (modulo_11) dime parte por parte que (ojo retire el archivo o carpeta tools que tenias antes)





# 05 08 2026
# queaceres
# 0-  se debe scrapear (VI, SOT, PKATT, PKATTALLOW)
# 1-  se debe actualizar (funcion_tabla_desempegno) para afectar variable opcionales con metricas abstractas (VI, SOT, PKATT, PKATTALLOW) tras cada partido
# 2-  se debe actualizar (calculo_metricas_0) para aceptar el drop manual de metricas base de los usuarios
# 3-  se debe configurar la DB en settings.py
# 4-  se debe hacer el migrate para crear tablas en la DB postgresql
# 5-  se debe actualizar el sistema para usar decoradores en el agregado de tablas de mejores terceros por mundiales
# 6-  se debe verificar que prediccionees historicas si sean base de inferencia en fase de interes
# 7-  se debe actualizar el frontend para tomar elecciones de metricas abstractas y drop de metricas asi como el envio de estas por api fetch
# 8-  se debe actualizar las views.py para tomar las elecciones de metricas abstractas y drop de metricas y enviarlas bien al (modulo_i.py)
# 9-  se debe actualizar (modulo_i.py) para aceptar insersion de metricas abstractas en (OPTIONAL_METRICS) y metricas a dropear en (DROP_METRICS)
# 10-)  funciones a comentar o recomentar: 
#     10.0-)  (fase_de_grupos)
#     10.1-)  (jugar)
#     10.2-)  (funcion_tabla_desempegno)
#     10.3-)  (formar_dataset_real)
#     10.4-)  (calc_gkps)
#     10.5-)  (calc_mds)
#     10.6-)  (calc_mos)
#     10.7-)  (calc_mms)
#     10.8-)  (calc_rate)
#     10.9-)  (obtener_diferencia)
#     10.10-)  (calculo_metricas_0)

# Aclaraciones
# - lo que sale de (grupos_anio_interes) no afecta a metricas fase a fase de df2 o df3 que usa (seleccionar) solo se emplean en (fase_de_grupos) en sus campos (pais,Pts) 
#   para definir el lugar de cada equipo segun su desempeño
# 
# estructura de columnas con la que se entrenara el modelo
home   PTS_0   PJ_0   PG_0   PP_0   PE_0   gkps_0   mds_0   mos_0   mms_0   rate_GC_0   rate_GF_0   D_0   away   PTS_1   PJ_1   PG_1   PP_1   PE_1   gkps_1   mds_1   mos_1   mms_1   rate_GC_1   rate_GF_1   D_1   

# (PROMPTS)
# muestrame aqui una version de la funcion (funcion_tabla_desempegno) del archivo (modulo_11.py) y actualizala agregandole la logica del script de (C:\Users\USUARIO\Trabajo\proyectos\mundiales\futbolInferencia\03 09 2026 0\scripts\imputar_opcionales.py). NO ACTUALICES NADA EN EL ARCHIVO ORIGINAL SOLO MUESTRAME LA VERSION QUE PIDO Y SI SE REQUIERE ACTUALIZAR ALGO EN EL SCRIPT (modulo_11) dime parte por parte que (ojo retire el archivo o carpeta tools que tenias antes)


