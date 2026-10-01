import numpy as np
import pandas as pd

def calculo_metricas_0():
    pass

def calculo_metricas_1():
    pass

def fase_de_grupos(self, dic_t, df_fixture_, model):
    # 1. Lookup table: equipo -> nombre del grupo
    grupo_por_equipo = pd.Series({
        equipo.strip(): nombre_grupo
        for nombre_grupo, df_grupo in dic_t.items()
        for equipo in df_grupo["pais"]
    })

    # 2. Copia de trabajo con el grupo localizado para cada fila
    trabajo = df_fixture_.copy()
    serieBomboA = pd.Series(trabajo["home"].str.strip().map(grupo_por_equipo))
    serieBomboB = pd.Series(trabajo["away"].str.strip().map(grupo_por_equipo))

    # 3. Series finales de goles alineadas al índice original
    score_0 = pd.Series(index=df_fixture_.index, dtype=float)
    score_1 = pd.Series(index=df_fixture_.index, dtype=float)

    # Subset limpio, sin la columna auxiliar _grupo
    bombos = pd.DataFrame({'bhome': serieBomboA, 'baway': serieBomboB})

    # Lógica de inferencia (misma idea que en modulo_11.py)
    if self.lista_fases.index(self.ind0[0]) < self.lista_fases.index(self.ind0[1]):
        puntos = self.intsc_prob_goles(np.array([]), model, alterno=1)
    else:
        df_fix_group_n = calculo_metricas_0(df_fix_group_n)
        X = df_fix_group_n.to_numpy()
        puntos = self.intsc_prob_goles(X, model, alterno=0)

    # 5. Actualizar puntos y guardar marcadores alineados al índice original
    for idx, points_, row in zip(df_fixture_.index, puntos, df_fix_group_n.itertuples()):
        home = row.home.strip()
        away = row.away.strip()

        # Actualización del grupo por equipo
        dic_t.loc[dic_t["pais"] == home, "Pts"] += points_[0][1]
        dic_t.loc[dic_t["pais"] == away, "Pts"] += points_[1][1]

        # Guardamos scores en la posición original de cada partido
        score_0.at[idx] = points_[0][0]
        score_1.at[idx] = points_[1][0]

    # 6. Ordenar, limpiar y redondear el grupo actualizado
        dic_t[bombos.iloc[idx, 'home']] = (
            dic_t
            .sort_values("Pts", ascending=False)
            .reset_index(drop=True)
            .loc[:, ["pais", "Pts"]]
            .round(0)
        )

        dic_t[bombos.iloc[idx, 'away']] = (
            dic_t
            .sort_values("Pts", ascending=False)
            .reset_index(drop=True)
            .loc[:, ["pais", "Pts"]]
            .round(0)
        )

    # 7. Escribir los marcadores en el fixture original
    df_fixture_["score_0"] = score_0
    df_fixture_["score_1"] = score_1
    calculo_metricas_1(df_fixture_)

    return dic_t