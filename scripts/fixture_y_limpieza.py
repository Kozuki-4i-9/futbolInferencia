import pandas as pd
import os

os.chdir("C:\\Users\\Usuario\\Trabajo\\python\\data\\p_mundiales\\partidos\\")

# es cuestionable la edicion sobre las columnas "Unnamed: 0" pero debemos crear en este archivo la logica
# del fixture.

# -----------------------------------------------------------------------
# limpiamos el archivo de partidos mundiales de columnas no deseadas

df0 = pd.read_csv("partidos_mundiales.csv")
df0 = df0.drop(["Unnamed: 0"], axis=1)
df0.to_csv("partidos_mundiales.csv")

# -----------------------------------------------------------------------
# aqui dejaremos la logica de obtencion del data fixture.
# VA A TARDAR

df1 = pd.read_csv("partidos_2022.csv")

df1.iloc[:,1] = ""
df1.iloc[48:,0] = ""
df1.iloc[48:,2] = ""

df1.to_csv("partidos_2022_fixture.csv")