from selenium import webdriver
from fake_useragent import UserAgent

from selenium.webdriver import Firefox
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By

import numpy as np
import pandas as pd

years = [1934, 1938]
dir_base = "C:\\Users\\Usuario\\Trabajo\\python\\p_mundiales\\partidos\\"

# 1934 1938

def get_teams_and_matches(year):
    web = f"https://es.wikipedia.org/wiki/Copa_Mundial_de_F%C3%BAtbol_de_{year}"
    dir_gdr = "C:\\Users\\Usuario\\Trabajo\\python\\data\\p_mundiales\\partidos"

    useragent = UserAgent()
    options=Options()
    options.set_preference('profile', useragent.random)
    options.binary_location = r"C:\\Program Files\\Mozilla Firefox\\firefox.exe"
    service = Service(r"C:\\Users\\Usuario\\Trabajo\\python\\controladores\\geckodriver.exe")
    driver = Firefox(service=service, options=options)
    driver.implicitly_wait(7)
    driver.maximize_window()
    driver.get(web)

    partidos_0 = []

    encontrados_0 = driver.find_elements(By.XPATH, "//table[@class='collapsible autocollapse vevent plainlist mw-collapsible mw-made-collapsible mw-collapsed']//tbody//tr//td//a[contains(@title, 'Selección de fútbol')]")
    encontrados_1 = driver.find_elements(By.XPATH, "//table[@class='collapsible autocollapse vevent plainlist mw-collapsible mw-made-collapsible mw-collapsed']//tbody//tr//td//b")

    if year=="1934":
        print(1, len(encontrados_1))
        encontrados_0 = [des0 for des0 in encontrados_0]
        encontrados_1 = [des1 for des1 in encontrados_1][::2]
        print(2, len(encontrados_1))
    elif year=="1938":
        encontrados_0 = [des0 for des0 in encontrados_0][:-1]
        encontrados_1 = [des0 for des0 in encontrados_1]
        encontrados_1.insert(-15,encontrados_1[-15])
        encontrados_1 = encontrados_1[::2]

    #print(encontrados_1)

    for des0, des1, des2 in zip(encontrados_0[0::2], encontrados_1, encontrados_0[1::2]):
        partidos_0.append([des0.text.strip(), des1.text.strip()[:3], des2.text.strip()])

    print("partidos_0:\n\n")
    print(partidos_0)
    print("\n\n")

    df_partidos = pd.DataFrame(data=np.array(partidos_0), columns=["home","score","away"])
    df_partidos["year"] = year
    df_partidos.to_csv(dir_gdr+f"\\partidos_{year}.csv")
    
    return(df_partidos)

for year in years:
    df_equipos_partidos_historico = get_teams_and_matches(str(year))
    print("df_equipos_partidos_historico:\n\n")
    print(df_equipos_partidos_historico)
    print("\n\n")