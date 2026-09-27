import os
import pandas as pd
import numpy as np

# Rutas de archivos
csv_path = r"c:\Users\USUARIO\Trabajo\proyectos\mundiales\futbolInferencia\data\international_matches.csv"
partidos_csv = r"c:\Users\USUARIO\Trabajo\proyectos\mundiales\futbolInferencia\data\partidos\partidos.csv"
partidos_1930_csv = r"c:\Users\USUARIO\Trabajo\proyectos\mundiales\futbolInferencia\data\partidos\partidos_1930.csv"
desktop_dir = r"C:\Users\USUARIO\Desktop\metricas_rendimiento_01072026700"

print("Cargando el archivo de partidos internacionales...")
df_im = pd.read_csv(csv_path)

# Asegurar orden cronológico de partidos internacionales
df_im['date'] = pd.to_datetime(df_im['date'])
df_im = df_im.sort_values('date').reset_index(drop=True)

# Calcular 'away_team_result' para los partidos internacionales
def get_away_result(row):
    home_res = row['home_team_result']
    if home_res == 'Win':
        return 'Lose'
    elif home_res == 'Lose':
        return 'Win'
    else:
        return 'Draw'

df_im['away_team_result'] = df_im.apply(get_away_result, axis=1)

# Calcular 'home_team_avg_fifa_points' y 'away_team_avg_fifa_points' en todo el dataset internacional
print("Calculando el promedio acumulado histórico de puntos FIFA por equipo en el dataset internacional...")
team_points_history = {}
home_avg_pts = []
away_avg_pts = []

for idx, row in df_im.iterrows():
    home_team = row['home_team']
    away_team = row['away_team']
    home_pts = row['home_team_total_fifa_points']
    away_pts = row['away_team_total_fifa_points']
    
    # Promedio acumulado para equipo local
    if home_team in team_points_history and len(team_points_history[home_team]) > 0:
        home_avg = np.mean(team_points_history[home_team])
    else:
        home_avg = home_pts
        team_points_history[home_team] = []
        
    # Promedio acumulado para equipo visitante
    if away_team in team_points_history and len(team_points_history[away_team]) > 0:
        away_avg = np.mean(team_points_history[away_team])
    else:
        away_avg = away_pts
        team_points_history[away_team] = []
        
    home_avg_pts.append(home_avg)
    away_avg_pts.append(away_avg)
    
    # Registrar en historial
    team_points_history[home_team].append(home_pts)
    team_points_history[away_team].append(away_pts)

df_im['home_team_avg_fifa_points'] = home_avg_pts
df_im['away_team_avg_fifa_points'] = away_avg_pts

# Guardar todos los partidos internacionales con promedios en el Escritorio
if not os.path.exists(desktop_dir):
    os.makedirs(desktop_dir)
todos_partidos_out = os.path.join(desktop_dir, "todos_partidos_metricas.csv")
df_im.to_csv(todos_partidos_out, index=False)
print(f"Guardado: {todos_partidos_out} ({len(df_im)} partidos)")

# ----------------- PROCESAMIENTO DE MUNDIALES (1930 - 2022) -----------------
print("\nCargando y uniendo los partidos de todos los mundiales (1930 - 2022)...")

# Cargar partidos de todos los mundiales
df_1930 = pd.read_csv(partidos_1930_csv, encoding='utf-8')
if df_1930.columns[0] == 'Unnamed: 0' or df_1930.columns[0] == '':
    df_1930 = df_1930.iloc[:, 1:]

df_others = pd.read_csv(partidos_csv, encoding='utf-8')
df_wc_raw = pd.concat([df_1930, df_others], ignore_index=True)

# Diccionario de traducción Español -> Inglés
es_to_en = {
    'Alemania': 'Germany', 'España': 'Spain', 'Italia': 'Italy', 'Brasil': 'Brazil',
    'Suecia': 'Sweden', 'Francia': 'France', 'Países Bajos': 'Netherlands',
    'Argentina': 'Argentina', 'Uruguay': 'Uruguay', 'Inglaterra': 'England',
    'Bélgica': 'Belgium', 'Croatia': 'Croatia', 'Croacia': 'Croatia', 'Portugal': 'Portugal',
    'México': 'Mexico', 'Colombia': 'Colombia', 'Chile': 'Chile',
    'Japón': 'Japan', 'Corea del Sur': 'Korea Republic', 'Estados Unidos': 'USA',
    'Suiza': 'Switzerland', 'Dinamarca': 'Denmark', 'Paraguay': 'Paraguay',
    'Perú': 'Peru', 'Ecuador': 'Ecuador', 'Costa Rica': 'Costa Rica',
    'Camerún': 'Cameroon', 'Nigeria': 'Nigeria', 'Marruecos': 'Morocco',
    'Senegal': 'Senegal', 'Arabia Saudita': 'Saudi Arabia', 'Rusia': 'Russia',
    'Sudáfrica': 'South Africa', 'Turquía': 'Turkey', 'Grecia': 'Greece',
    'Polonia': 'Poland', 'República Checa': 'Czech Republic', 'Rumania': 'Romania',
    'Bulgaria': 'Bulgaria', 'Ucrania': 'Ukraine', 'Irán': 'Iran',
    'Australia': 'Australia', 'Egipto': 'Egypt', 'Ghana': 'Ghana',
    'Costa de Marfil': "Côte d'Ivoire", 'Túnez': 'Tunisia', 'Argelia': 'Algeria',
    'Islandia': 'Iceland', 'Panamá': 'Panama', 'Honduras': 'Honduras',
    'Nueva Zelanda': 'New Zealand', 'Eslovaquia': 'Slovakia', 'Eslovenia': 'Slovenia',
    'Serbia': 'Serbia', 'Angola': 'Angola', 'Togo': 'Togo', 'Trinidad y Tobago': 'Trinidad and Tobago',
    'Jamaica': 'Jamaica', 'China': 'China PR', 'Irlanda': 'Republic of Ireland',
    'Irlanda del Norte': 'Northern Ireland', 'Gales': 'Wales', 'Escocia': 'Scotland',
    'Austria': 'Austria', 'Noruega': 'Norway', 'Hungría': 'Hungary',
    'Checoslovaquia': 'Czechoslovakia', 'Yugoslavia': 'Yugoslavia', 'Unión Soviética': 'Soviet Union',
    'Alemania Democrática': 'German DR', 'Alemania Federal': 'Germany',
    'Zaire': 'Zaire', 'Haití': 'Haiti', 'El Salvador': 'El Salvador',
    'Kuwait': 'Kuwait', 'Irak': 'Iraq', 'Canadá': 'Canada',
    'Bolivia': 'Bolivia', 'Venezuela': 'Venezuela', 'Catar': 'Qatar', 'Bosnia y Herzegovina': 'Bosnia and Herzegovina',
    'República Democrática del Congo': 'DR Congo', 'Corea del Norte': 'Korea DPR',
    'Emiratos Árabes Unidos': 'United Arab Emirates',
    'Indias Orientales Neerlandesas': 'Dutch East Indies', 'Israel': 'Israel', 'Cuba': 'Cuba'
}

# Inverso para traducción Inglés -> Español
en_to_es = {v: k for k, v in es_to_en.items()}
# Correcciones manuales inversas
en_to_es['Germany'] = 'Alemania'
en_to_es['USA'] = 'Estados Unidos'
en_to_es['Netherlands'] = 'Países Bajos'
en_to_es['Republic of Ireland'] = 'Irlanda'
en_to_es['Korea Republic'] = 'Corea del Sur'
en_to_es['Korea DPR'] = 'Corea del Norte'
en_to_es['United Arab Emirates'] = 'Emiratos Árabes Unidos'
en_to_es['DR Congo'] = 'República Democrática del Congo'
en_to_es['Dutch East Indies'] = 'Indias Orientales Neerlandesas'
en_to_es['Israel'] = 'Israel'
en_to_es['Cuba'] = 'Cuba'

# Mapeo de correcciones para partidos históricos y de 2022 con definición por penales (shootouts)
# Clave: (home_team_es, away_team_es, year) -> (home_score_real, away_score_real, shoot_out_flag, home_team_result, away_team_result)
shootout_corrections = {
    # 1982
    ('Alemania', 'Francia', 1982): (3, 3, 'Yes', 'Win', 'Lose'),
    # 1986
    ('Brasil', 'Francia', 1986): (1, 1, 'Yes', 'Lose', 'Win'),
    ('Alemania', 'México', 1986): (0, 0, 'Yes', 'Win', 'Lose'),
    ('España', 'Bélgica', 1986): (1, 1, 'Yes', 'Lose', 'Win'),
    # 1990
    ('Irlanda', 'Rumania', 1990): (0, 0, 'Yes', 'Win', 'Lose'),
    ('Yugoslavia', 'Argentina', 1990): (0, 0, 'Yes', 'Lose', 'Win'),
    ('Italia', 'Argentina', 1990): (1, 1, 'Yes', 'Lose', 'Win'),
    ('Alemania', 'Inglaterra', 1990): (1, 1, 'Yes', 'Win', 'Lose'),
    # 2022
    ('Japón', 'Croacia', 2022): (1, 1, 'Yes', 'Lose', 'Win'),
    ('Croacia', 'Brasil', 2022): (1, 1, 'Yes', 'Win', 'Lose'),
    ('Países Bajos', 'Argentina', 2022): (2, 2, 'Yes', 'Lose', 'Win'),
    ('Marruecos', 'España', 2022): (0, 0, 'Yes', 'Win', 'Lose'),
    ('Argentina', 'Francia', 2022): (3, 3, 'Yes', 'Win', 'Lose')
}

# Añadir columnas al DataFrame mundial de destino
cols_wc = [
    'year', 'date', 'home_team', 'away_team',
    'home_team_fifa_rank', 'away_team_fifa_rank',
    'home_team_total_fifa_points', 'away_team_total_fifa_points',
    'home_team_avg_fifa_points', 'away_team_avg_fifa_points',
    'home_team_score', 'away_team_score', 'shoot_out',
    'home_team_result', 'away_team_result',
    'home_team_goalkeeper_score', 'away_team_goalkeeper_score',
    'home_team_mean_defense_score', 'home_team_mean_offense_score', 'home_team_mean_midfield_score',
    'away_team_mean_defense_score', 'away_team_mean_offense_score', 'away_team_mean_midfield_score'
]

# Crear dataframe vacío para almacenar los partidos procesados
processed_wc = []

print("Procesando y enriqueciendo cada partido de mundial...")
unmapped_teams = set()

# Extraer el año de los partidos internacionales para facilitar coincidencia
df_im['year'] = df_im['date'].dt.year

for idx, row in df_wc_raw.iterrows():
    home_es = row['home'].strip()
    away_es = row['away'].strip()
    year = int(row['year'])
    score_raw = str(row['score']).strip()
    
    # Traducir a inglés
    home_en = es_to_en.get(home_es, home_es)
    away_en = es_to_en.get(away_es, away_es)
    
    if home_es not in es_to_en:
        unmapped_teams.add(home_es)
    if away_es not in es_to_en:
        unmapped_teams.add(away_es)
        
    # Inicializar campos básicos
    date_val = None
    shoot_out = 'No'
    home_team_result = 'Draw'
    away_team_result = 'Draw'
    
    # Analizar marcador estándar
    try:
        parts = score_raw.split(':')
        home_score = int(parts[0])
        away_score = int(parts[1])
    except:
        home_score = 0
        away_score = 0
        
    # Definir resultado por defecto basado en goles
    if home_score > away_score:
        home_team_result = 'Win'
        away_team_result = 'Lose'
    elif home_score < away_score:
        home_team_result = 'Lose'
        away_team_result = 'Win'
        
    # Aplicar correcciones manuales de penales si el partido está en la lista
    if (home_es, away_es, year) in shootout_corrections:
        home_score, away_score, shoot_out, home_team_result, away_team_result = shootout_corrections[(home_es, away_es, year)]
    elif (away_es, home_es, year) in shootout_corrections:
        # Si están invertidos local y visitante en la clave del diccionario
        away_score, home_score, shoot_out, away_team_result, home_team_result = shootout_corrections[(away_es, home_es, year)]

    # Inicializar métricas avanzadas como nulas
    adv_metrics = {
        'home_team_fifa_rank': np.nan, 'away_team_fifa_rank': np.nan,
        'home_team_total_fifa_points': np.nan, 'away_team_total_fifa_points': np.nan,
        'home_team_avg_fifa_points': np.nan, 'away_team_avg_fifa_points': np.nan,
        'home_team_goalkeeper_score': np.nan, 'away_team_goalkeeper_score': np.nan,
        'home_team_mean_defense_score': np.nan, 'home_team_mean_offense_score': np.nan, 'home_team_mean_midfield_score': np.nan,
        'away_team_mean_defense_score': np.nan, 'away_team_mean_offense_score': np.nan, 'away_team_mean_midfield_score': np.nan
    }

    # Intentar buscar coincidencia en international_matches.csv para mundiales entre 1994 y 2018
    if 1994 <= year <= 2018:
        # Filtrar por año y torneo Copa del Mundo
        df_match_candidates = df_im[(df_im['year'] == year) & (df_im['tournament'] == 'FIFA World Cup')]
        
        # Buscar coincidencia exacta de equipos (en inglés)
        match_row = df_match_candidates[
            ((df_match_candidates['home_team'] == home_en) & (df_match_candidates['away_team'] == away_en)) |
            ((df_match_candidates['home_team'] == away_en) & (df_match_candidates['away_team'] == home_en))
        ]
        
        if len(match_row) > 0:
            match_data = match_row.iloc[0]
            date_val = match_data['date']
            shoot_out = match_data['shoot_out']
            
            # Si el equipo local en partidos.csv es el local en international_matches.csv
            if match_data['home_team'] == home_en:
                home_score = match_data['home_team_score']
                away_score = match_data['away_team_score']
                home_team_result = match_data['home_team_result']
                away_team_result = match_data['away_team_result']
                
                # Cargar métricas avanzadas en orden normal
                for col in adv_metrics.keys():
                    if col in match_data:
                        adv_metrics[col] = match_data[col]
            else:
                # Si los roles están invertidos
                home_score = match_data['away_team_score']
                away_score = match_data['home_team_score']
                home_team_result = match_data['away_team_result']
                away_team_result = match_data['home_team_result']
                
                # Cargar métricas avanzadas e invertirlas
                adv_metrics['home_team_fifa_rank'] = match_data['away_team_fifa_rank']
                adv_metrics['away_team_fifa_rank'] = match_data['home_team_fifa_rank']
                adv_metrics['home_team_total_fifa_points'] = match_data['away_team_total_fifa_points']
                adv_metrics['away_team_total_fifa_points'] = match_data['home_team_total_fifa_points']
                adv_metrics['home_team_avg_fifa_points'] = match_data['away_team_avg_fifa_points']
                adv_metrics['away_team_avg_fifa_points'] = match_data['home_team_avg_fifa_points']
                
                adv_metrics['home_team_goalkeeper_score'] = match_data['away_team_goalkeeper_score']
                adv_metrics['away_team_goalkeeper_score'] = match_data['home_team_goalkeeper_score']
                
                adv_metrics['home_team_mean_defense_score'] = match_data['away_team_mean_defense_score']
                adv_metrics['home_team_mean_offense_score'] = match_data['away_team_mean_offense_score']
                adv_metrics['home_team_mean_midfield_score'] = match_data['away_team_mean_midfield_score']
                
                adv_metrics['away_team_mean_defense_score'] = match_data['home_team_mean_defense_score']
                adv_metrics['away_team_mean_offense_score'] = match_data['home_team_mean_offense_score']
                adv_metrics['away_team_mean_midfield_score'] = match_data['home_team_mean_midfield_score']

    # Estructura del registro procesado (versión inglés)
    record_en = {
        'year': year,
        'date': date_val if date_val is not None else f"{year}-06-15", # fecha aproximada para partidos sin coincidencia
        'home_team': home_en,
        'away_team': away_en,
        'home_team_score': home_score,
        'away_team_score': away_score,
        'shoot_out': shoot_out,
        'home_team_result': home_team_result,
        'away_team_result': away_team_result,
        **adv_metrics
    }
    processed_wc.append(record_en)

# Imprimir advertencia si hay selecciones sin traducción
if len(unmapped_teams) > 0:
    print(f"¡Atención! Equipos no mapeados en traducción: {unmapped_teams}")

# Crear DataFrames finales
df_wc_en = pd.DataFrame(processed_wc)

# Crear la versión en español traduciendo de vuelta los nombres de equipos
df_wc_es = df_wc_en.copy()
df_wc_es['home_team'] = df_wc_es['home_team'].map(lambda x: en_to_es.get(x, x))
df_wc_es['away_team'] = df_wc_es['away_team'].map(lambda x: en_to_es.get(x, x))

# Guardar los archivos de mundial en el Escritorio
out_en_path = os.path.join(desktop_dir, "partidos_mundiales_completos_en.csv")
out_es_path = os.path.join(desktop_dir, "partidos_mundiales_completos_es.csv")

df_wc_en.to_csv(out_en_path, index=False, encoding='utf-8')
df_wc_es.to_csv(out_es_path, index=False, encoding='utf-8')

print(f"Guardado: {out_en_path} ({len(df_wc_en)} partidos)")
print(f"Guardado: {out_es_path} ({len(df_wc_es)} partidos)")

print("\n¡Proceso de extracción completado con éxito!")
