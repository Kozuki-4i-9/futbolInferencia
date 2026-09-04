# DF1 (equipos): columnas ['VI', 'SOT', 'PKATT', 'PKATTALLOW'] por equipo (totales acumulados hasta antes de la jornada).

# DF2 (partidos): columnas con estadísticas de home y away (goles, puntos, etc.). Contiene los resultados de la jornada que acaba de jugarse.

for idx, row in DF2.iterrows():
    home = row['home_team']
    away = row['away_team']
    GF_home = row['GF_home']
    GF_away = row['GF_away']
    
    # --- VI ---
    if GF_away == 0:
        DF1.loc[away, 'VI'] += 1
    if GF_home == 0:
        DF1.loc[home, 'VI'] += 1
    
    # --- Obtener promedios históricos previos ---
    # (usamos el total acumulado / partidos jugados)
    home_avg_sot = DF1.loc[home, 'SOT'] / max(1, DF1.loc[home, 'matches_played'])
    away_avg_sot = DF1.loc[away, 'SOT'] / max(1, DF1.loc[away, 'matches_played'])
    
    home_avg_pkatt = DF1.loc[home, 'PKATT'] / max(1, DF1.loc[home, 'matches_played'])
    away_avg_pkatt = DF1.loc[away, 'PKATT'] / max(1, DF1.loc[away, 'matches_played'])
    
    # --- Imputar SOT con ajuste por goles ---
    # Si metió más goles de lo normal, ajustamos al alza
    home_gf_ratio = GF_home / max(0.5, home_avg_sot * 0.3)  # ratio de eficiencia
    away_gf_ratio = GF_away / max(0.5, away_avg_sot * 0.3)
    
    DF1.loc[home, 'SOT'] += round(home_avg_sot * (0.7 + 0.3 * min(home_gf_ratio, 2)), 1)
    DF1.loc[away, 'SOT'] += round(away_avg_sot * (0.7 + 0.3 * min(away_gf_ratio, 2)), 1)
    
    # --- Imputar PKATT (pocos cambios, casi constante) ---
    DF1.loc[home, 'PKATT'] += round(home_avg_pkatt * 0.9, 2)
    DF1.loc[away, 'PKATT'] += round(away_avg_pkatt * 0.9, 2)
    
    # --- PKATTALLOW (similar a PKATT) ---
    # Asumimos que concede similar a lo que ataca
    DF1.loc[home, 'PKATTALLOW'] += round(away_avg_pkatt * 0.9, 2)  # concede del rival
    DF1.loc[away, 'PKATTALLOW'] += round(home_avg_pkatt * 0.9, 2)
    
    # --- Actualizar partidos jugados ---
    DF1.loc[home, 'matches_played'] += 1
    DF1.loc[away, 'matches_played'] += 1