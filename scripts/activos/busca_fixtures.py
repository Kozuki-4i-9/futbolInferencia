import pandas as pd
import numpy as np

dir_base = "C:\\Users\\Usuario\\Trabajo\\python\\data\\p_mundiales\\grupos\\grupos_mundiales.csv"

class busca_fixture():
    def __init__(self, desafios, df_gen, primer_desafio, anio, mod=0):
        pd0 = pd.read_csv(dir_base)

        pd0.rename(columns={"Unnamed: 0":"pais"}, inplace=True)

        pd1 = pd0[["pais", "Grupo", "year"]]
        self.dic_anio = {}

#           {1950:{"Fase Final":(6)}}
        dic_desf_0 = {
            1930:{"semi":(2,0,1),"final":(1,0,1)},
            1934:{"8vos":(8,0,1),"4tos":(5,0,1),"semi":(2,0,1),"3er":(1,0,1),"final":(1,1,2)},
            1938:{"8vos":(10,0,1),"4tos":(5,0,1),"semi":(2,0,1),"3er":(1,0,1),"final":(1,1,2)},
            1950:{"Fase Final":(6)},
            1954:{"4tos":(4,0,1),"semi":(2,0,1),"3er":(1,0,1),"final":(1,1,2)},
            1958:{"4tos":(4,0,1),"semi":(2,0,1),"3er":(1,0,1),"final":(1,1,2)},
            1962:{"4tos":(4,0,1),"semi":(2,0,1),"3er":(1,0,1),"final":(1,1,2)},
            1966:{"4tos":(4,0,1),"semi":(2,0,1),"3er":(1,0,1),"final":(1,1,2)},
            1970:{"4tos":(4,0,1),"semi":(2,0,1),"3er":(1,0,1),"final":(1,1,2)},
            1974:{"Fase Inicial":(12),"3er":(1,0,1),"final":(1,1,2)},
            1978:{"Fase Inicial":(12),"3er":(1,0,1),"final":(1,1,2)},
            1982:{"Fase Inicial":[12],"semi":(2,0,1),"3er":(1,0,1),"final":(1,1,2)},
            1986:{"8vos":(8,0,1),"4tos":(4,0,1),"semi":(2,0,1),"3er":(1,0,1),"final":(1,1,2)},
            1990:{"8vos":(8,0,1),"4tos":(4,0,1),"semi":(2,0,1),"3er":(1,0,1),"final":(1,1,2)},
            1994:{"8vos":(8,0,1),"4tos":(4,0,1),"semi":(2,0,1),"3er":(1,0,1),"final":(1,1,2)},
            1998:{"8vos":(8,0,1),"4tos":(4,0,1),"semi":(2,0,1),"3er":(1,0,1),"final":(1,1,2)},
            2002:{"8vos":(8,0,1),"4tos":(4,0,1),"semi":(2,0,1),"3er":(1,0,1),"final":(1,1,2)},
            2006:{"8vos":(8,0,1),"4tos":(4,0,1),"semi":(2,0,1),"3er":(1,0,1),"final":(1,1,2)},
            2010:{"8vos":(8,0,1),"4tos":(4,0,1),"semi":(2,0,1),"3er":(1,0,1),"final":(1,1,2)},
            2014:{"8vos":(8,0,1),"4tos":(4,0,1),"semi":(2,0,1),"3er":(1,0,1),"final":(1,1,2)},
            2018:{"8vos":(8,0,1),"4tos":(4,0,1),"semi":(2,0,1),"3er":(1,0,1),"final":(1,1,2)},
            2022:{"8vos":(8,0,1),"4tos":(4,0,1),"semi":(2,0,1),"3er":(1,0,1),"final":(1,1,2)},
            2026:{"16vos":(16,0,1),"8vos":(8,0,1),"4tos":(4,0,1),"semi":(2,0,1),"3er":(1,0,1),"final":(1,1,2)}
            }

        dic_lugar = {0:"Winner", 1:"Runners-up", 2:"Third-p", 3:"Fourth-p"}

        self.df_count_nueva_fase = df_gen[:primer_desafio] # lleva el registro de partidos de la nueva fase

        if not anio in (1934, 1938):
            for year in pd1.year.unique().tolist():
                dic_gpd2 = {}
                pd2 = pd1[pd1["year"]==year]
                for grupo in pd2.Grupo.unique().tolist():
                    dic_gpd2[grupo] = pd2.loc[pd2["Grupo"]==grupo, "pais"].to_numpy().tolist()
                self.dic_anio[year] = dic_gpd2

        for ndesf, desf0 in enumerate(desafios):
            df_partidos = pd.DataFrame()

            if desf0 == "3er":
                situacion = "Loser "
            else:
                situacion = "Winner "

            if (ndesf==0 or anio in (1950, 1974, 1978)) or (anio in (1982,) and ndesf in (0,1,)) and not anio in (1934, 1938):
                # iteramos sobre todos los partidos del primer desafio tras acabar la fase de grupos
                # puede ser 8vos, 4tos, 3rlugar, semi o final.

                dic_items = self.dic_anio[anio].items()

                if anio == 1950 or anio in (1974, 1978, 1982):
                    # Dejo los n "Grupo i" en "dic_items" (pares clave-valor de entre los grupos
                    # del año)
                    if(anio == 1950):
                        # Me quedo con todos los grupos menos con "Fase Final"
                        tamagno = df_gen[:primer_desafio]
                        dic_items = [(des[0], des[1],) for des in self.dic_anio[anio].items()][:-1]
                        desafio_actual = df_gen[primer_desafio:primer_desafio + dic_desf_0[anio][desf0]]
                    elif(anio in (1974, 1978,) and ndesf==0):
                        # Me quedo solo con "Grupo A" y "Grupo B"
                        tamagno = df_gen[:primer_desafio]
                        dic_items = [(des[0], des[1],) for des in self.dic_anio[anio].items() if not des[0]in("Grupo A","Grupo B")]
                        desafio_actual = df_gen[primer_desafio:primer_desafio + dic_desf_0[anio][desf0]]
                    elif(anio in (1982,) and ndesf==0):
                        print("dic0")
                        tamagno = df_gen[:primer_desafio]
                        dic_items = [(des[0], des[1],) for des in self.dic_anio[anio].items() if not des[0]in("Grupo A","Grupo B","Grupo C","Grupo D")]
                        desafio_actual = df_gen[primer_desafio:primer_desafio + dic_desf_0[anio][desf0][0]]
                    elif(anio in (1982,) and ndesf==1):
                        # Me quedo solo con "Grupo A" y "Grupo B"
                        tamagno = df_gen[:len(self.df_count_nueva_fase)]
                        dic_items = [(des[0], des[1],) for des in self.dic_anio[anio].items() if des[0]in("Grupo A","Grupo B","Grupo C","Grupo D")]
                        print("dic1")
                        print(dic_items)
                        desafio_actual = df_gen[len(self.df_count_nueva_fase):len(self.df_count_nueva_fase) + dic_desf_0[anio][desf0][0]]
                    elif(anio in (1974, 1978,) and ndesf>0):
                        # Me quedo solo con "Grupo A" y "Grupo B"
                        tamagno = df_gen[:len(self.df_count_nueva_fase)]
                        dic_items = [(des[0], des[1],) for des in self.dic_anio[anio].items() if des[0]in("Grupo A","Grupo B")]
                        desafio_actual = df_gen[len(self.df_count_nueva_fase):len(self.df_count_nueva_fase) + dic_desf_0[anio][desf0][0]]
                    
                    # dic_desf_0: define la estructura del mundial de cada "anio" y "desf0" indica
                    # cada fase del mundial, el valor "[0]" indica el tamaño de paso de dicha fase

                if(anio not in (1950, 1974, 1978, 1982)):
                    tamagno = df_gen[:primer_desafio]
                    print("-0-", dic_desf_0[anio][desf0], dic_desf_0[anio][desf0])
                    desafio_actual = df_gen[primer_desafio:primer_desafio + dic_desf_0[anio][desf0][0]]

                for desf2, desf3 in zip(desafio_actual, range(1, len(desafio_actual) + 1)):
                    npartido = "Match {}".format(str(len(tamagno) + desf3))
                    print(f"0-dic_items: {len(desafio_actual)}\n{dic_items}")
                    print(f"0-desafio_actual: {len(desafio_actual)}\n{desafio_actual}\n{desf2}")
                    print("")
                    print("")
                    for grp in dic_items: # dic_items: grupos del mundial iterados
                        if desf2[0] in grp[1]: # desf2: se iteran los partidos de la fase actual
                            g1 = grp[0] # nombre del grupo "Fase Final"
                            ng1 = grp[1].index(desf2[0]) # lugar que ocupo en el grupo
                        if desf2[2] in grp[1]:
                            g2 = grp[0] # nombre del grupo "Fase Final"
                            ng2 = grp[1].index(desf2[2]) # lugar que ocupo en el grupo

                    partido = pd.DataFrame(np.array([[f"{dic_lugar[ng1]} {g1}", npartido, f"{dic_lugar[ng2]} {g2}", anio]]), columns=["home", "score", "away", "year"])
                    df_partidos = pd.concat([df_partidos, partido], axis=0)

            elif (ndesf>0 and anio not in (1974, 1978, 1950)) or anio in (1934, 1938) and not(anio in (1982,) and ndesf==1):

                tamagno = self.df_count_nueva_fase
                if not anio in (1934, 1938) or ndesf>0:
                    guia0 = self.df_count_nueva_fase
                elif anio in (1934, 1938) and ndesf==0:
                    guia0 = []
                desafio_actual = df_gen[len(guia0):len(guia0) + dic_desf_0[anio][desf0][0]]

                for desf2, desf3 in zip(desafio_actual, range(1, len(desafio_actual) + 1)):
                    # desf2: lista de partidos de la fase actual
                    # desf3: del 1 a n segun sea el tamagno de la fase actual siendo n dicho tamagno
                    npartido = "Match {}".format(str(len(tamagno) + desf3))

                    llaves = [des for des in dic_desf_0[anio].keys()]
                    # llaves: ["16vos", "8vos", "4tos", "semi", "3er", "final"]
                    ind0 = dic_desf_0[anio][llaves[llaves.index(desf0) - dic_desf_0[anio][desf0][2]]][0] # semi: 5
                    # el desafio  actual  "desf0" nos  dice con su valor en "dic_desf_1" cuantos desafios
                    # retroceder en las "llaves" (si estamos en "final" retrocedemos del indice de final,
                    # 2 valores a "semi").
                    ind1 = dic_desf_0[anio][desf0][1] # semi: 0
                    # df_gen es un numpy
                    # En los partidos del mundial "df_gen"
                    if ndesf>0:
                        partidos_anteriores_1 = self.df_count_nueva_fase[-(ind0 + ind1):].to_numpy()
                        partidos_anteriores_0 = df_gen[len(self.df_count_nueva_fase) - (ind0 + ind1):len(self.df_count_nueva_fase) - ind1]

                        # partidos de interes del df fixture generado (df_count_nueva_fase) para tomar el
                        # numero del partido.

                        # print("partidos_anteriores_0:\n\n{}\n\npartidos_anteriores_1:\n\n{}".format(partidos_anteriores_0, partidos_anteriores_1))
                        # print("desf2[0]:\n\n{}\n\ndesf2[2]:\n\n{}".format(desf2[0], desf2[2]))

                        m1 = [match_1[1] for match_0, match_1 in zip(partidos_anteriores_0, partidos_anteriores_1) if desf2[0] in match_0][0]
                        m2 = [match_1[1] for match_0, match_1 in zip(partidos_anteriores_0, partidos_anteriores_1) if desf2[2] in match_0][0]

                        partido = pd.DataFrame(np.array([[f"{situacion}{m1}", npartido, f"{situacion}{m2}", anio]]), columns=["home", "score", "away", "year"])
                        df_partidos = pd.concat([df_partidos, partido], axis=0)

                    elif anio in (1934, 1938) and ndesf==0:

                        partido = pd.DataFrame(np.array([[desf2[0], npartido, desf2[2], anio]]), columns=["home", "score", "away", "year"])
                        df_partidos = pd.concat([df_partidos, partido], axis=0)

                # print("df_partidos:\n\n{}\n".format(df_partidos))
                # print("df_count_nueva_fase 1:\n\n{}".format(self.df_count_nueva_fase))

            self.df_count_nueva_fase = np.concatenate((self.df_count_nueva_fase, df_partidos), axis=0)
            self.df_count_nueva_fase = pd.DataFrame(data=self.df_count_nueva_fase, columns=["home","score","away","year"])

        print("df_count_nueva_fase: 2\n\n{}".format(self.df_count_nueva_fase))
        print("")
        print("")
        print("")

        self.df_count_nueva_fase.to_csv("C:\\Users\\Usuario\\Trabajo\\python\\data\\p_mundiales\\partidos\\fixture_1982.csv")

dir_gen = "C:\\Users\\Usuario\\Trabajo\\python\\data\\p_mundiales\\partidos\\partidos.csv"
dir_gen = pd.read_csv(dir_gen)
print(dir_gen)
dir_gen = dir_gen[dir_gen.year==str(1982)]
df_gen = dir_gen.to_numpy()

fix = busca_fixture(("Fase Inicial","semi","3er","final"), df_gen, 36, 1982)

#"16vos","8vos","4tos","semi","3er","final"
# se eliminan partidos de desempate para 1934 y 1938