# =====================================================================
#  EMPAREJAMIENTO POR MEJORES TERCEROS (MUNDIALES 1986, 1990 Y 1994)
# =====================================================================

_POSICIONES = {"1": "Winner", "2": "Runners-up", "3": "Third-p"}

# Reglamento FIFA (el mismo en los 3 mundiales): segun que 4 grupos aporten mejor tercero,
# de que grupo es el tercero que enfrenta a cada ganador de los grupos A, B, C y D
_CRUCES_TERCEROS = MappingProxyType({
  frozenset(grupos): dict(zip("ABCD", rivales))
  for grupos, rivales in {
    "ABCD": "CDAB", "ABCE": "CABE", "ABCF": "CABF", "ABDE": "DABE", "ABDF": "DABF",
    "ABEF": "EABF", "ACDE": "CDAE", "ACDF": "CDAF", "ACEF": "CAFE", "ADEF": "DAFE",
    "BCDE": "CDBE", "BCDF": "CDBF", "BCEF": "ECBF", "BDEF": "EDBF", "CDEF": "CDFE",
  }.items()
})

# Orden de juego del 1er round de la fase final. Notacion "<posicion><grupo>": "2E" = Runners-up Grupo E.
# "1A-3" = el ganador del grupo A enfrenta al mejor tercero que le asigne _CRUCES_TERCEROS
_LLAVES_1986_1994 = MappingProxyType({
  1986: ("1A-3", "1F-2E", "2A-2C", "1B-3", "1C-3", "2F-2B", "1E-2D", "1D-3"),
  1990: ("1A-3", "1E-2D", "2F-2B", "1B-3", "2A-2C", "1C-3", "1D-3", "1F-2E"),
  1994: ("2A-2C", "1A-3", "1B-3", "1F-2E", "1C-3", "1E-2D", "1D-3", "2F-2B"),
})

def _etiqueta(codigo):
  """Traduce un codigo de llave ("2E") a la etiqueta que entiende "camb_grp_elm_smpl" ("Runners-up Grupo E")"""
  return f"{_POSICIONES[codigo[0]]} Grupo {codigo[1]}"

def tabla_terceros(dict_table, df_grupos=None):
  """
  ¿QUE HACE?
  Genera un df con el 3er lugar de cada grupo ordenado segun el criterio FIFA: puntos, diferencia de gol y goles a favor. La letra del grupo se usa como ultimo desempate para que el resultado sea determinista

  ¿COMO LO HACE?
  0-) Se toma la fila de indice 2 (3er lugar) de cada tabla de (dict_table), ya ordenada por "fase_de_grupos"

  1-) Si se entrega (df_grupos), se acumulan los goles a favor y en contra de cada equipo como home y away, y se agregan las columnas "DG" y "GF" a los terceros; si no, valen 0

  2-) Se ordena de mejor a peor tercero

  Args:

    - dict_table: diccionario de salida de "fase_de_grupos" con clave "Grupo X" y valor el df (pais, Pts) ordenado

    - df_grupos: df de la fase de grupos ya jugada con columnas "home", "away", "score_0" y "score_1" (opcional, para desempatar)

  Returns:

    - df con columnas "Grupo", "pais", "Pts", "DG", "GF" ordenado de mejor a peor tercero
  """
  terceros = pd.DataFrame(
    [(grupo[-1], tabla["pais"].iloc[2], tabla["Pts"].iloc[2]) for grupo, tabla in dict_table.items()],
    columns=["Grupo", "pais", "Pts"]).astype({"Pts": float})

  if df_grupos is not None:
    columnas = ["pais", "GF", "GC"]
    goles = pd.concat([df_grupos[["home", "score_0", "score_1"]].set_axis(columnas, axis=1),
                       df_grupos[["away", "score_1", "score_0"]].set_axis(columnas, axis=1)])
    goles = goles.astype({"GF": float, "GC": float}).groupby("pais").sum()
    terceros = terceros.join(goles.assign(DG=goles.GF - goles.GC)[["DG", "GF"]], on="pais")
  terceros = terceros.reindex(columns=["Grupo", "pais", "Pts", "DG", "GF"]).fillna({"DG": 0, "GF": 0})

  return terceros.sort_values(["Pts", "DG", "GF", "Grupo"], ascending=[False, False, False, True], ignore_index=True)





def armar_llave(agno, grupos_terceros):
  """Devuelve el df (home, away) del 1er round de la fase final para (agno), dados los grupos de los 4 mejores terceros"""
  rival_de = _CRUCES_TERCEROS[frozenset(grupos_terceros)]
  partidos = []
  for partido in _LLAVES_1986_1994[agno]:
    local, visita = partido.split("-")
    if visita == "3":
      visita += rival_de[local[1]]
    partidos.append((_etiqueta(local), _etiqueta(visita)))
  return pd.DataFrame(partidos, columns=["home", "away"])

def alinear_knockout_1986_1994(dict_table, agno, df_grupos=None):
  """Alinea el 1er round de la fase final a partir de las tablas de grupos ya jugadas. Deja los 4 mejores terceros en attrs["mejores_terceros"]"""
  terceros = tabla_terceros(dict_table, df_grupos).iloc[:4]
  llave = armar_llave(agno, terceros.Grupo)
  llave.attrs["mejores_terceros"] = terceros
  return llave





@dataclass(frozen=True)
class ReglamentoTerceros:
  """Reglamento de mejores terceros de un mundial: orden de la llave y tabla de cruces 1ro vs 3ro"""
  agno: int
  llave: tuple
  cruces: MappingProxyType

def con_reglamento_terceros(func):
  """
  ¿QUE HACE?
  Decorador que recibe el año del mundial, valida que use el formato de mejores terceros y entrega a la funcion decorada el (ReglamentoTerceros) de ese año en lugar del año. La funcion decorada solo se ocupa de alinear equipos con las tablas de grupos ya jugadas
  """
  @wraps(func)
  def envoltura(dict_table, agno, *args, **kwargs):
    if agno not in _LLAVES_1986_1994:
      raise ValueError(f"El mundial {agno} no usa el formato de mejores terceros (solo 1986, 1990 y 1994)")
    reglamento = ReglamentoTerceros(agno, _LLAVES_1986_1994[agno], _CRUCES_TERCEROS)
    return func(dict_table, reglamento, *args, **kwargs)
  return envoltura

@con_reglamento_terceros
def alinear_knockout_1986_1994(dict_table, reglamento, df_grupos=None):
  """
  ¿QUE HACE?
  Genera el df (home, away) del 1er round de la fase final segun los 4 mejores terceros de las tablas de grupos ya jugadas y el (reglamento) inyectado por "con_reglamento_terceros"

  Args:

    - dict_table: salida de "fase_de_grupos"

    - reglamento: (ReglamentoTerceros) inyectado por el decorador; quien llama pasa el año (int)

    - df_grupos: df de la fase de grupos jugada, para desempatar por diferencia de gol y goles a favor

  Returns:

    - df con columnas "home" y "away" en formato "Winner Grupo A" / "Third-p Grupo C", con los 4 mejores terceros en attrs["mejores_terceros"]
  """
  terceros = tabla_terceros(dict_table, df_grupos).iloc[:4]
  rival_de = reglamento.cruces[frozenset(terceros.Grupo)]
  partidos = []
  for local, visita in (partido.split("-") for partido in reglamento.llave):
    if visita == "3":
      visita += rival_de[local[1]]
    partidos.append((_etiqueta(local), _etiqueta(visita)))
  llave = pd.DataFrame(partidos, columns=["home", "away"])
  llave.attrs["mejores_terceros"] = terceros
  return llave
