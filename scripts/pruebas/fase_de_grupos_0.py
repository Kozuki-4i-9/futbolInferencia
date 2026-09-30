import numpy as np
import pandas as pd

def calculo_metricas_0():
    pass

def calculo_metricas_1():
    pass

def fase_de_grupos(self, dic_t, df_fixture_, model):
    x = np.array([])
    X = np.array([])
    grupoH, grupoA = None, None
    grupos = pd.DataFrame()

    for _, row in df_fixture_.itertuples():
        dic_temp = {'home':None, 'away':None}
        
        for i, equipo in enumerate((row.home.strip(), row.away.strip())):
            for group, df in dic_t.items():
                if((equipo in df['pais'].values) and (i==0)):
                    dic_temp['home'] = group
                elif((equipo in df['pais'].values) and (i==1)):
                    dic_temp['away'] = group
        
        grupos = pd.concat([grupos, pd.DataFrame([dic_temp])], axis=0, ignore_index=True)

    if self.lista_fases.index(self.ind0[0]) < self.lista_fases.index(self.ind0[1]):
        puntos = self.intsc_prob_goles(X, model, alterno=1)
    else:
        X = df_fix_group_n.copy().to_numpy()
        df_fix_group_n = calculo_metricas_0(df_fix_group_n)
        puntos = self.intsc_prob_goles(X, model, alterno=0)

    for points_, idx, row in zip(puntos, df_fix_group_n.itertuples()):
        home, away = row.home, row.away
        x = np.append(x, [points_[0][0], points_[1][0]])
        
        grupoH, grupoA = dic_t[grupos.loc[idx, 'home']], dic_t[grupos.loc[idx, 'away']]

        grupoH.loc[grupoH['pais'] == home, 'Pts'] += points_[0][1]
        grupoA.loc[grupoA['pais'] == away, 'Pts'] += points_[1][1]

    grupoH, grupoA = grupoH.sort_values('Pts', ascending=False).reset_index(drop=True), grupoA.sort_values('Pts', ascending=False).reset_index(drop=True)
    grupoH, grupoA = grupoH[['pais', 'Pts']], grupoA[['pais', 'Pts']]
    grupoH, grupoA = grupoH.round(0), grupoA.round(0)

    df_fixture_["score_0"] = x[0::2]
    df_fixture_["score_1"] = x[1::2]

    calculo_metricas_1(df_fixture_)

    return dic_t