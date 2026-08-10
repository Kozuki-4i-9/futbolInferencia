from fake_useragent import UserAgent

from selenium.webdriver import Firefox
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By

import numpy as np
import pandas as pd

web = f"https://es.wikipedia.org/wiki/Copa_Mundial_de_F%C3%BAtbol_de_2014"

useragent = UserAgent()
options=Options()
options.set_preference('profile', useragent.random)
service = Service(r'C:\\Users\\user\\Trabajo\\python\\controladores\\geckodriver.exe')
driver = Firefox(service=service, options=options)
driver.get(web)

base = driver.find_element(By.XPATH, "//a[contains(@title, 'Selección de fútbol de Alemania')]")

data_grupo_0 = base.find_elements(By.XPATH, "//preceding::table[@style='background-color: #f5faff; border: 1px #aaa solid; border-collapse: collapse; font-size: 95%; margin: 36px auto; text-align: center;']//tbody")

lst_col_tr0 = np.array([])
cabecera_t = []

# me itero sobre todas (8) las tablas
for des0, des1 in enumerate(data_grupo_0):
    data_grupo_ = des1.find_elements(By.XPATH, ".//child::tr")
    # tr de la cabecera
    data_grupo_10 = data_grupo_[0]
    # tr para los grupos
    data_grupo_11 = data_grupo_[1:]

    if des0==0:
        for des2, des3 in enumerate(data_grupo_10.find_elements(By.XPATH, ".//child::th")):
            if des2==0:
                cabecera_t.append(des3.text.strip())
            elif des2>0:
                agregacion = des3.find_element(By.XPATH, ".//abbr")
                cabecera_t.append(agregacion.text.strip())
        print("cabecera: ", cabecera_t)
        cabecera_t.append("Grupo")

    # me itero sobre los grupos y sus caracteristicas para imprimir y 
    # meterlose en el dataframe.
    for des4 in data_grupo_11:
        data_grupo_2 = des4.find_elements(By.XPATH, './/child::td')
        lst_col_tr1 = []
        # //a[contains(@title, 'Selección de fútbol de Alemania')]//preceding::table[@style='background-color: #f5faff; border: 1px #aaa solid; border-collapse: collapse; font-size: 95%; margin: 36px auto; text-align: center;']//tbody//child::tr//preceding::h3[1]//span[contains(@id, 'Grupo_')][1]
        confirm_grupo = des4.find_elements(By.XPATH, ".//preceding::h3//span[contains(@id, 'Grupo_')][1]")

        for des5, des6 in enumerate(data_grupo_2):
            if des5==0:
                impresion = des6.find_element(By.XPATH, ".//a[contains(@title, 'Selección de fútbol de')]").text.strip()
                lst_col_tr1.append(impresion)
            elif des5==1:
                impresion = des6.find_element(By.XPATH, ".//b").text.strip()
                lst_col_tr1.append(impresion)
            elif des5>1:
                impresion = des6.text.strip()
                lst_col_tr1.append(impresion)

        lst_col_tr1.append(confirm_grupo[des0].text.strip())
        lst_col_tr0 = np.append(lst_col_tr0, lst_col_tr1)

lst_col_tr0 = np.reshape(lst_col_tr0, (int(len(lst_col_tr0)/10),10))

print(cabecera_t)
print(lst_col_tr0[:,0])

#df_base = pd.DataFrame(data=lst_col_tr0, columns=cabecera_t)
df_base = pd.DataFrame(data=lst_col_tr0[:,1:], columns=cabecera_t[1:], index=lst_col_tr0[:,0])

df_base["year"] = "2014"
df_base.to_csv("grupos_mundiales_2014.csv")