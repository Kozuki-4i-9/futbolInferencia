# ctrl + k + c 

# Tomamos la version ordenada oringinal y la desordenada de los grupos de cada mundial y
# obtenemos, de cada uno respectivamente, un archivo general con los grupos de todos los
# mundiales en el orden descargado original y en la version desordenada.

# 0 desordenar_grupos_y_partidos.py ---> 1 grupos_partidos_mundiales.py

import numpy as np
import pandas as pd

years = [1930, 1934, 1938, 1950, 1954, 1958, 1962, 1966, 1970, 1974, 1978, 1982, 1986, 1990, 1994, 1998, 2002, 2006, 2010, 2014, 2018, 2022]
dir_base = "C:\\Users\\Usuario\\Trabajo\\python\\data\\p_mundiales\\grupos\\"
dir_pg = ""
dfcol = None

for des00, des01 in enumerate(["grupos_mundiales_","partidos_"]):
    for des1 in range(2):
        df_base = pd.DataFrame()
        for year in years:
            if des00==0 and des1==0 and not year==1934 and not year==1938: # obtendremos el total ordenado y desordenado de los grupos
                dir_pg = dir_base + des01+f"{year}.csv"
                mis_equipos_partidos = pd.read_csv(dir_pg)
                dfcol = pd.read_csv(dir_base + des01 + "2014.csv")
                col0 = list(dfcol.columns[1:])
                final_1 = ".csv"

                df1 = pd.DataFrame(data=mis_equipos_partidos.iloc[:,1:].to_numpy(), columns=col0,
                                index=mis_equipos_partidos.iloc[:,0])
                df_base = pd.concat([df_base, df1], axis=0)

            elif des00==0 and des1==1 and not year==1934 and not year==1938:
                dir_pg = dir_base + des01 + f"{year}_mo.csv"
                mis_equipos_partidos = pd.read_csv(dir_pg)
                dfcol = pd.read_csv(dir_base+des01 + "2014.csv")
                col0 = list(dfcol.columns[1:])
                final_1 = "_mo.csv"

                df1 = pd.DataFrame(data=mis_equipos_partidos.iloc[:,1:].to_numpy(), columns=col0,
                                index=mis_equipos_partidos.iloc[:,0])
                df_base = pd.concat([df_base, df1], axis=0)

            if des00==1 and des1==0: # obtenemos el total ordenado y desordenado de los partidos
                dir_pg = dir_base.replace("grupos\\",f"partidos\\") + f"{des01}"+f"{year}.csv"
                mis_equipos_partidos = pd.read_csv(dir_pg)
                dfcol = pd.read_csv(dir_base.replace("grupos\\",f"partidos\\") + des01 + "2014.csv")
                col0 = list(dfcol.columns[1:])
                final_1 = ".csv"

                df1 = pd.DataFrame(data=mis_equipos_partidos.iloc[:,1:].to_numpy(), columns=col0,
                                index=mis_equipos_partidos.iloc[:,0])
                df_base = pd.concat([df_base, df1], axis=0)

            elif des00==1 and des1==1:
                dir_pg = dir_base.replace("grupos\\",f"partidos\\") + f"{des01}{year}_mo.csv"
                mis_equipos_partidos = pd.read_csv(dir_pg)
                dfcol = pd.read_csv(dir_base.replace("grupos\\",f"partidos\\") + des01 + "2014.csv")
                col0 = list(dfcol.columns[1:])
                final_1 = "_mo.csv"

                df1 = pd.DataFrame(data=mis_equipos_partidos.iloc[:,1:].to_numpy(), columns=col0,
                                index=mis_equipos_partidos.iloc[:,0])
                df_base = pd.concat([df_base, df1], axis=0)

        n0 = dir_pg[::-1].find("\\")
        
        a = dir_pg[:len(dir_pg)-(n0 + 1)] + "\\" + des01[:len(des01)-1] + final_1
        df_base.to_csv(a)
        