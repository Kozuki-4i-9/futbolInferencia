# directrices de entrenamiento basado en el archivo (modulo_i.py)

# el modelo se entreno con los datos de los partidos de todos los mundiales (947 row hasta 2022) con esta estructura de inputs:

# home   PTS_0   PG_0   PP_0   PE_0   D_0   score_0   ts_GF_0   ts_GC_0   away   PTS_1   PG_1   PP_1   PE_1   D_1   score_1   ts_GF_1   ts_GC_1

# x, y = calculo_metricas_0(df1), df1[["score_0", "score_1"]] 

# se le aplica una capa Embedding de tensorflow
# x0 = objs.hacer_embedding_a_equipos(data_0) # home   -   x1 = objs.hacer_embedding_a_equipos(data_1) # away

# se le aplica una funcion para tomar lotes de entreda tamaño (n) y la correspondiente salida (modelo timestep para LSTMs)

# x_tr_h, x_tr_a = objs.c_s(x0[:607]), objs.c_s(x1[:607])
# xtr_home, xtr_away = x_tr_h[0], x_tr_a[0]
# ytr_home, ytr_away = x_tr_h[1], x_tr_a[1]
# X_tr = np.concatenate((xtr_home, xtr_away), axis=0)
# Y_tr = np.concatenate((ytr_home, ytr_away), axis=0) # PEND probar si podemos hacer shuffle
# X_tr.shape, Y_tr.shape   ->  (1204, 5, 10) (1204,)

# el modelo usado hasta ahora, probado y aprobado el diciembre de 2024 es:

# model = tf.keras.Sequential([
#     tf.keras.layers.LSTM(100, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
#     tf.keras.layers.Dropout(0.20),
#     tf.keras.layers.LSTM(57, return_sequences=True),
#     tf.keras.layers.Dropout(0.40),
#     tf.keras.layers.LSTM(25, return_sequences=False),
#     tf.keras.layers.Dropout(0.15),
#     tf.keras.layers.Dense(32, activation='relu'),
#     tf.keras.layers.Dense(1)  # Salida para regresión
# ])
# # Compilar el modelo
# model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mae'])

# model.fit(X_tr, Y_tr,  batch_size=16, validation_data=(X_val, Y_val), epochs=110, callbacks=[TensorMetricas]) 





# 30 09 2026

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
# 11-) se debe trabajar o idear la adaptacion a Champions League y Copa Libertadores de America, para que el sistema pueda predecir resultados de estas competiciones con la misma logica de fases y metricas abstractas

# Aclaraciones
# - lo que sale de (grupos_anio_interes) no afecta a metricas fase a fase de df2 o df3 que usa (seleccionar) solo se emplean en (fase_de_grupos) en sus campos (pais,Pts) 
#   para definir el lugar de cada equipo segun su desempeño
# 
# estructura de columnas con la que se entrenara el modelo

# home   PTS_0   PJ_0   PG_0   PP_0   PE_0   gkps_0   mds_0   mos_0   mms_0   rate_GC_0   rate_GF_0   D_0   away   PTS_1   PJ_1   PG_1   PP_1   PE_1   gkps_1   mds_1   mos_1   mms_1   rate_GC_1   rate_GF_1   D_1   

# (PROMPTS)
# muestrame aqui una version de la funcion (funcion_tabla_desempegno) del archivo (modulo_11.py) y actualizala agregandole la logica del script de (C:\Users\USUARIO\Trabajo\proyectos\mundiales\futbolInferencia\03 09 2026 0\scripts\imputar_opcionales.py). NO ACTUALICES NADA EN EL ARCHIVO ORIGINAL SOLO MUESTRAME LA VERSION QUE PIDO Y SI SE REQUIERE ACTUALIZAR ALGO EN EL SCRIPT (modulo_11) dime parte por parte que (ojo retire el archivo o carpeta tools que tenias antes)





# 06 10 2026 0

# ── Cambios Pendientes:
#       7. Cambiar la contraseña en PostgreSQL
#       Si cambiaste DB_PASSWORD en el .env pero no en la base de datos, Django no podrá conectarse. En psql, como superusuario (postgres):
#       ALTER USER mundial WITH PASSWORD 'una-contraseña-nueva-fuerte';
#       Tiene que ser exactamente la misma que pusiste en .env.

