# web/fases/modulo_11.py
import pandas as pd

BASE_METRICS, ALL_METRICS, OPTIONAL_METRICS = None, None, None

def formar_dataset_real(): # PEND recomentar docstring
    pass

def funcion_tabla_desempegno(df0, pais, indx=None, agno=None, ind0=0):
    global opcionales

    dic_dsp = {}
    dic_dsp["year"] = agno
    dic_dsp["pais"] = pais

    primero = pd.DataFrame()

    suffix_0 = "_0" if ind0 == 1 else ""
    suffix_1 = "_1" if ind0 == 1 else ""

    col_map_0 = {m: f"{m}{suffix_0}" for m in ALL_METRICS}
    col_map_1 = {m: f"{m}{suffix_1}" for m in ALL_METRICS}

    pts_0_col, pj_0_col, pg_0_col, pp_0_col, pe_0_col, gf_0_col, gc_0_col, d_0_col = \
        [col_map_0[m] for m in BASE_METRICS]
    pts_1_col, pj_1_col, pg_1_col, pp_1_col, pe_1_col, gf_1_col, gc_1_col, d_1_col = \
        [col_map_1[m] for m in BASE_METRICS]

    # --- preparación de insumos históricos para imputar opcionales ---
    insumos = None
    if ind0 == 1 and opcionales is None:
        try: # PEND: evaluar si realmente es esto necesario
            opcionales = formar_dataset_real("proxy_desempegno", agno)
        except Exception: # PEND: evaluar si realmente es esto necesario
            opcionales = None # PEND: evaluar si realmente es esto necesario

    if ind0 == 1 and opcionales is not None and "pais" in opcionales.columns:
        insumos = opcionales.set_index("pais")
        if "PJ" not in insumos.columns and "matches_played" in insumos.columns:
            insumos["PJ"] = insumos["matches_played"] # PEND: evaluar si realmente es esto necesario

    for indice, partido in enumerate(df0.copy().itertuples(index=False)):
        if partido.home == pais:
            if primero.empty:
                init_vals = {col_map_0[m]: 0 for m in BASE_METRICS}
                for m in OPTIONAL_METRICS:
                    init_vals[col_map_0[m]] = 0
                dic_dsp.update(init_vals)

            # ... aquí va tu lógica base de PG/PP/PE/PTS/GF/GC ...
            dic_dsp[d_0_col] = dic_dsp[gf_0_col] - dic_dsp[gc_0_col]
            dic_dsp[pj_0_col] += 1
            primero = pd.DataFrame(data=dic_dsp, index=[0])

            # --- imputación de opcionales para el país como HOME ---
            if ind0 == 1 and insumos is not None:
                gf_pais = partido.score_0
                gf_rival = partido.score_1

                sot_previo = insumos.at[pais, "SOT"]
                pkatt_previo = insumos.at[pais, "PKATT"]
                pkatt_previo_rival = insumos.at[partido.away, "PKATT"]

                pj_previo = insumos.at[pais, "PJ"]
                pj_previo_rival = insumos.at[partido.away, "PJ"]

                if "VI" in col_map_0:
                    dic_dsp[col_map_0["VI"]] += 1 if gf_rival == 0 else 0

                if "SOT" in col_map_0:
                    avg_sot = sot_previo / max(1, pj_previo)
                    ratio = gf_pais / max(0.5, avg_sot * 0.3)
                    dic_dsp[col_map_0["SOT"]] += round(
                        avg_sot * (0.7 + 0.3 * min(ratio, 2)), 1
                    )

                if "PKATT" in col_map_0:
                    dic_dsp[col_map_0["PKATT"]] += round(
                        pkatt_previo / max(1, pj_previo) * 0.9, 2
                    )

                if "PKATTALLOW" in col_map_0:
                    dic_dsp[col_map_0["PKATTALLOW"]] += round(
                        pkatt_previo_rival / max(1, pj_previo_rival) * 0.9, 2
                    )

            if ind0 == 1:
                cols_to_update = [col_map_0[m] for m in ALL_METRICS]
                df0.loc[indice, cols_to_update] = pd.Series(dic_dsp)

        elif partido.away == pais:
            if primero.empty:
                init_vals = {col_map_1[m]: 0 for m in BASE_METRICS}
                for m in OPTIONAL_METRICS:
                    init_vals[col_map_1[m]] = 0
                dic_dsp.update(init_vals)

            # ... aquí va tu lógica base de PG/PP/PE/PTS/GF/GC ...
            dic_dsp[d_1_col] = dic_dsp[gf_1_col] - dic_dsp[gc_1_col]
            dic_dsp[pj_1_col] += 1
            primero = pd.DataFrame(data=dic_dsp, index=[0])

            # --- imputación de opcionales para el país como AWAY ---
            if ind0 == 1 and insumos is not None:
                gf_pais = partido.score_1
                gf_rival = partido.score_0

                sot_previo = insumos.at[pais, "SOT"]
                pkatt_previo = insumos.at[pais, "PKATT"]
                pkatt_previo_rival = insumos.at[partido.home, "PKATT"]

                pj_previo = insumos.at[pais, "PJ"]
                pj_previo_rival = insumos.at[partido.home, "PJ"]

                if "VI" in col_map_1:
                    dic_dsp[col_map_1["VI"]] += 1 if gf_rival == 0 else 0

                if "SOT" in col_map_1:
                    avg_sot = sot_previo / max(1, pj_previo)
                    ratio = gf_pais / max(0.5, avg_sot * 0.3)
                    dic_dsp[col_map_1["SOT"]] += round(
                        avg_sot * (0.7 + 0.3 * min(ratio, 2)), 1
                    )

                if "PKATT" in col_map_1:
                    dic_dsp[col_map_1["PKATT"]] += round(
                        pkatt_previo / max(1, pj_previo) * 0.9, 2
                    )

                if "PKATTALLOW" in col_map_1:
                    dic_dsp[col_map_1["PKATTALLOW"]] += round(
                        pkatt_previo_rival / max(1, pj_previo_rival) * 0.9, 2
                    )

            if ind0 == 1:
                cols_to_update = [col_map_1[m] for m in ALL_METRICS]
                df0.loc[indice, cols_to_update] = pd.Series(dic_dsp)

    if ind0 == 0:
        return pd.DataFrame(dic_dsp, index=[indx])

def calculo_metricas_1(df_desempegno):
    global opcionales

    # ... cargar opcionales si hace falta ...

    # PRIMERA PASADA: imputar opcionales dentro de df_desempegno
    # AQUÍ df_desempegno todavía NO tiene los columnas opcionales imputadas
    paises = pd.concat([df_desempegno.home, df_desempegno.away], axis=0).unique()
    for indip, pais in enumerate(paises):
        funcion_tabla_desempegno(df_desempegno, pais, ind0=1)

    # SEGUNDA PASADA: df_desempegno YA pasó por funcion_tabla_desempegno
    # y ya contiene columnas como "VI_0", "SOT_0", ..., "VI_1", "SOT_1", ...
    for _, partido in df_desempegno.iterrows():
        home = partido.home
        away = partido.away

        for metrica in OPTIONAL_METRICS:
            opcionales.loc[opcionales.pais == home, metrica] += partido[f"{metrica}_0"]
            opcionales.loc[opcionales.pais == away, metrica] += partido[f"{metrica}_1"]

    return df_desempegno