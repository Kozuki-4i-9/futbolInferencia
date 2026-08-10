from fake_useragent import UserAgent

from selenium.webdriver import Firefox
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By

import numpy as np
import pandas as pd

web = f"https://es.wikipedia.org/wiki/Copa_Mundial_de_F%C3%BAtbol_de_1958"
        
useragent = UserAgent()
options=Options()
options.set_preference('profile', useragent.random)
service = Service(r"C:\\Users\\Usuario\\Trabajo\\python\\controladores\\geckodriver.exe")
driver = Firefox(service=service, options=options)
driver.get(web)

base = driver.find_element(By.XPATH, "//a[contains(@title, 'Selección de fútbol de')]")

data_grupo_0 = base.find_elements(By.XPATH, "//preceding::table[@style='background: #f9f9f9; border: 1px #aaa solid; border-collapse: collapse; font-size: 95%;' and @cellpadding='3']//tbody")

print("data_grupo_0: ", len(data_grupo_0))

lst_col_tr0 = np.array([])
cabecera_t = []

# me itero sobre todas (8) las tablas
for des0, des1 in enumerate(data_grupo_0[:]):
    data_grupo_ = des1.find_elements(By.XPATH, ".//child::tr")
    # tr de la cabecera
    data_grupo_10 = data_grupo_[0]
    # tr para los grupos
    data_grupo_11 = data_grupo_[1:]
    confirm_grupo = des1.find_elements(By.XPATH, f".//preceding::div[contains(@class, 'mw-heading mw-heading4')]//h4[contains(@id, 'Grupo_')]")
# //a[contains(@title, 'Selección de fútbol de')]//preceding::table[@style='background: #f9f9f9; border: 1px #aaa solid; border-collapse: collapse; font-size: 95%;' and @cellpadding='3']//tbody//preceding::div[contains(@class, 'mw-heading mw-heading4')]//h4[contains(@id, 'Grupo_')]
    print("data_grupo_10: ", len(data_grupo_11))

    if des0==0:
        for des2, des3 in enumerate(data_grupo_10.find_elements(By.XPATH, ".//child::th")):
            cabecera_t.append(des3.text.strip())
        cabecera_t.append("Grupo")
        cabecera_t[0] = "Selección"
        print("cabecera_t: ", cabecera_t)

    # me itero sobre los grupos y sus caracteristicas para imprimir y 
    # meterlose en el dataframe.
    for des4 in data_grupo_11:
        data_grupo_2 = des4.find_elements(By.XPATH, './/child::td')
        lst_col_tr1 = []

        for des5, des6 in enumerate(data_grupo_2):
            if des5==0:
                impresion = des6.find_element(By.XPATH, ".//a[contains(@title, 'Selección de fútbol de') or @title='Corea del Sur']").text.strip()
                lst_col_tr1.append(impresion)
            elif des5==1:
                impresion = des6.find_element(By.XPATH, ".//b").text.strip()
                lst_col_tr1.append(impresion)
            elif des5>1:
                impresion = des6.text.strip()
                lst_col_tr1.append(impresion)

        lst_col_tr1.append(confirm_grupo[des0].text.strip())
        lst_col_tr0 = np.append(lst_col_tr0, lst_col_tr1)

        print("lst_col_tr0: ", lst_col_tr0)
 
lst_col_tr0 = np.reshape(lst_col_tr0, (int(len(lst_col_tr0)/10),10))

df_base = pd.DataFrame(data=lst_col_tr0[:,1:], columns=cabecera_t[1:], index=lst_col_tr0[:,0])

df_base["year"] = "1958"
dir_gen = "C:\\Users\\Usuario\\Trabajo\\python\\data\\p_mundiales\\grupos\\"
df_base.to_csv(dir_gen+"grupos_mundiales_1958.csv")