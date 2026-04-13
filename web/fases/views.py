from django.shortcuts import render
from django.http import JsonResponse
from .models import partidos, fixtures, grupos, clasificaciones
from django.core.serializers.json import DjangoJSONEncoder
import json
import pandas as pd
import numpy as np

from .modulo_0 import *

agnos = ["1934", "1938", "1950", "1954", "1958", "1962", "1966", "1970", "1974", "1978", "1982", "1986", "1990", "1994", "1998", "2002", "2006", "2010", "2014", "2018", "2026", "2022"]

def formar_dataset_real(ind0, valores=None):
    """
    Busca en la base de datos segun la tabla indicada por "ind0" y segun el año 
    presente en valores (no lo estará en el caso de ind0="clasificaciones").

    - Args:
            - ind0: cadena que indica a que tabla se hará la búsqueda
            - valores: año en que se hará la búsqueda (a no ser que sea un 
                       ind0="clasificaciones")

    - Returns:
               - df: dataframe con el contenido de la tabla que se ha consultado
    """
    if(ind0=="partidos"):
        df0 = list()
        for des in partidos.objects.all():
            if des.agno==valores:
                df0.append([des.home, des.score, des.away, des.agno])
        df = pd.DataFrame(data=np.array(df0), columns=["home", "score", "away", "year"])
        return(df)
    
    elif(ind0=="fixtures"):
        df0 = list()
        for des in fixtures.objects.all():
            if des.agno==valores:
                df0.append([des.home, des.score, des.away, des.agno])
        df = pd.DataFrame(data=np.array(df0), columns=["home", "score", "away", "year"])
        return(df)
    
    elif(ind0=="grupos"):
        df0 = list()
        for des in grupos.objects.all():
            if des.agno==valores:
                df0.append([des.pais, des.Pts, des.PJ, des.PG, des.PP, des.PE, des.GF, des.GC, des.Dif, des.Grupo, des.agno])
        df = pd.DataFrame(data=np.array(df0), columns=["pais", "Pts", "PJ", "PG", "PP", "PE", "GF", "GC", "Dif", "Grupo", "agno"])
        return(df)
    
    elif(ind0=="clasificaciones"):
        df0 = list()
        for des in clasificaciones.objects.all():
            df0.append([des.home, des.score_0, des.score_1, des.away, des.tournament, des.agno])
        df = np.array(df0)
        df = pd.DataFrame(data=df, columns=["home", "score_0", "score_1", "away", "tournament", "agno"])
        return(df)

def formar_dataset_consulta(valores):
    """
    Recive del template y arroja un df con formato adecuado para predecir

    - Args:
        - valores: lista de listas con datos para hacer un dataframe

    - Returns: 
        - dfd: dataframe con indices reiniciados y con columnas "home", "score_0", "score_1", "away"
    """
    dfd = pd.DataFrame()
    for val0, val1 in zip(valores[2:], range(len(valores[2:]))):
        print(f"val0\n{val0}")
        dfd = pd.concat([dfd, pd.DataFrame(data=np.array([[val0[0], int(val0[2]), int(val0[3]), val0[1]]]), columns=["home", "score_0", "score_1", "away"])], axis=0)
    dfd = dfd.reset_index(drop=True)#.iloc[:,1:]

    #print(f"dfd\n{dfd[["home", "score_0", "score_1", "away"]]}, {dfd.shape}")
    return(dfd)

dic_torneos = {"Fase de Grupos": "grupo", "Fase Inicial":"doce", "octavos": "knockout", "cuartos":"quarter", "semifinal":"semi", "3er_lugar":"third", "final": "final"}

def procesar_recepcion_0(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        opcion_0 = data.get('fecha', None)
        if(opcion_0 in agnos):
            df = formar_dataset_real("partidos", opcion_0)
            print(f"df:\n{df}")
            json_records_0 = df.to_json(orient='records')
            context = {'partidos_reales': json_records_0}
            print(f"Estos son mis partidos\n{df}")
            return(JsonResponse(context))

    return(render(request, "prediccion_2.html"))

def procesar_recepcion_1(request):
    """Vista usada al precionar el boton "predecir" """
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            recepcion = data.get('recepcion', None)
            if recepcion:

                recepc = formar_dataset_consulta(recepcion)
                if recepcion[1]=="Fase de Grupos":
                    recepc["agno"] = int(recepcion[0])

                respuesta = consulta_general(dic_torneos[recepcion[1]]+recepcion[0], recepc).reset_index(drop=True)
                respuesta.score_0 = respuesta.score_0.astype('int64')
                respuesta.score_1 = respuesta.score_1.astype('int64')
                score = pd.DataFrame()
                for des0, des1 in zip(respuesta.score_0, respuesta.score_1):
                    score = pd.concat([score, pd.Series(f"{des0}:{des1}")], axis=0)
                
                score = score.reset_index()[0]
                respuesta = respuesta.drop(["score_0", "score_1"], axis=1)
                print(respuesta)
                print(score)
                respuesta = pd.DataFrame(data={"home":respuesta.home, "score": score,"away":respuesta.away}).reset_index(drop=True)
                respuesta["year"]=int(recepcion[0])
                print(f"Esta es mi respuesta\n{respuesta}")

                json_partidos = respuesta.to_json(orient='records')

                context = {"partidos": json_partidos}

                return JsonResponse(context)
            
        except json.JSONDecodeError as e:
            print(f"Error al decodificar JSON: {e}")
 
            return JsonResponse({'error': 'JSON inválido'}, status=400)

    return JsonResponse({'error': 'Método no permitido'}, status=405)
