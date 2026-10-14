# ctrl + k + c

# generamos la version desrordenada de cada grupo.

# 0 desordenar_grupos_y_partidos.py ---> 1 grupos_partidos_mundiales.py

import pandas as pd
import numpy as np

years = [1930, 1934, 1938, 1950, 1954, 1958, 1962, 1966, 1970, 1974, 1978, 1982, 1986, 1990, 1994, 1998, 2002, 2006, 2010, 2014, 2018, 2022]
dir_base = "C:\\Users\\Usuario\\Trabajo\\python\\data\\p_mundiales\\grupos\\"

def desordenar0(dir_g):
    objetivos = pd.read_csv(dir_g)

    col_0 = list(objetivos.columns)

    objetivos = objetivos.to_numpy()

    indice_0 = np.arange(objetivos.shape[0])

    np.random.shuffle(indice_0); objetivos = objetivos[indice_0]

    dfg_0 = pd.DataFrame(data=objetivos[:,1:], columns=col_0[1:], index=objetivos[:,0])

    dfg_0.to_csv(dir_g.replace(".csv", "_mo.csv"))

def desordenar1(year, ind, dir=""):
    dir_g = ""
    if ind==0 and not year == "1934" and not year == "1938": # modificamos grupos
        dir_g = dir_base+f"grupos_mundiales_{year}.csv"
        desordenar0(dir_g)

    elif ind==1:
        dir_g = dir_base.replace("grupos\\","partidos\\")
        dir_g = dir_g + f"partidos_{year}.csv"
        desordenar0(dir_g)

    if dir!="":
        dir_g = dir
        desordenar0(dir_g)

for modo in range(2):
    for year in years:
        desordenar1(str(year), modo)