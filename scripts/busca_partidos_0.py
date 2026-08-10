from selenium import webdriver
from fake_useragent import UserAgent

from selenium.webdriver import Firefox
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By

import numpy as np
import pandas as pd

years = [1930, 1950, 1954, 1958, 1962, 1966, 1970, 1974, 1978, 1982, 1986, 1990, 1994, 1998, 2002, 2006, 2010, 2014, 2018, 2022]
dir_base = "C:\\Users\\Usuario\\Trabajo\\python\\data\\p_mundiales\\partidos\\"

# 2010 2014 

def get_teams_and_matches(year):
    web = f"https://es.wikipedia.org/wiki/Copa_Mundial_de_F%C3%BAtbol_de_{year}"
    dir_gdr = "C:\\Users\\Usuario\\Trabajo\\python\\data\\p_mundiales\\partidos"

    useragent = UserAgent()
    options=Options()
    options.set_preference('profile', useragent.random)
    options.binary_location = r"C:\\Program Files\\Mozilla Firefox\\firefox.exe"
    service = Service(r"C:\\Users\\Usuario\\Trabajo\\python\\data\\controladores\\geckodriver.exe")
    driver = Firefox(service=service, options=options)
    driver.implicitly_wait(7)
    driver.maximize_window()
    driver.get(web)

# //table[@class='collapsible autocollapse vevent plainlist mw-collapsible mw-made-collapsible mw-collapsed']//td//div[1]//b[1]
# //table[@class='collapsible autocollapse vevent plainlist mw-collapsible mw-made-collapsible mw-collapsed']//td
# //table[@class='collapsible autocollapse vevent plainlist mw-collapsible mw-made-collapsible mw-collapsed']//td//child::div

    partidos_0 = []
    save_0 = [0,0]

    encontrados_0 = driver.find_elements(By.XPATH, "//table[@class='collapsible autocollapse vevent plainlist mw-collapsible mw-made-collapsible mw-collapsed']//tbody//tr//td[@width='24%']//a[contains(@title, 'Selección de fútbol de') or contains(@title, 'Selección de Fútbol de')][1]")
    encontrados_1 = driver.find_elements(By.XPATH, "//table[@class='collapsible autocollapse vevent plainlist mw-collapsible mw-made-collapsible mw-collapsed']//td//child::div")
    encontrados_2 = driver.find_elements(By.XPATH, "//table[@class='collapsible autocollapse vevent plainlist mw-collapsible mw-made-collapsible mw-collapsed']//tbody//tr//td[@width='22%']//a[contains(@title, 'Selección de fútbol de') or contains(@title, 'Selección de Fútbol de')][1]")

    encontrado_0 = [des0.text.strip() for des0 in encontrados_0]
    encontrado_10 = [des0.text.strip() for des0 in encontrados_1]
    encontrado_2 = [des0.text.strip() for des0 in encontrados_2]

    for des0 in encontrado_10:
        if des0[0]=="(" and not des0[1:4]==partidos_0[-2]:
            partidos_0 = partidos_0[:-3]
            partidos_0.append(save_0[0])
            partidos_0.append(des0[1:4])
            partidos_0.append(save_0[1])
        elif des0[0] in ("1","2","3","4","5","6","7","8","9","0"):
            save_0[0] = encontrado_0[0]
            partidos_0.append(encontrado_0[0])
            encontrado_0.remove(encontrado_0[0])
            partidos_0.append(des0[:3])
            save_0[1] = encontrado_2[0]
            partidos_0.append(encontrado_2[0])
            encontrado_2.remove(encontrado_2[0])

    partidos_0 = np.reshape(np.array(partidos_0), (int(len(partidos_0)/3) ,3))
    df_partidos = pd.DataFrame(data=partidos_0, columns=["home","score","away"])
    df_partidos["year"] = year
    df_partidos.to_csv(dir_gdr+f"\\partidos_{year}.csv")
    
    return(df_partidos)

for year in years:
    df_equipos_partidos_historico = get_teams_and_matches(str(year))
    print(df_equipos_partidos_historico)