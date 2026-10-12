import re
from dataclasses import dataclass
from functools import wraps
from itertools import combinations
from types import MappingProxyType

# =====================================================================
#  EMPAREJAMIENTO POR MEJORES TERCEROS (MUNDIALES 1986, 1990 Y 1994)
# =====================================================================

_POSICIONES = {"1": "Winner", "2": "Runners-up", "3": "Third-p"}
_GRUPOS = "ABCDEF"
_GANADORES_CON_TERCERO = "ABCD"
_COMBINACIONES = frozenset(frozenset(c) for c in combinations(_GRUPOS, 4))
_CLASIFICADOS_FIJOS = frozenset(p + g for p in "12" for g in _GRUPOS)
_PATRON_PARTIDO = re.compile(r"[12][A-F]-(?:[12][A-F]|3)")

# Reglamento FIFA (el mismo en los 3 mundiales): segun que 4 grupos aporten mejor tercero,
# de que grupo es el tercero que enfrenta a cada ganador de los grupos A, B, C y D
_CRUCES_TERCEROS = MappingProxyType({
  frozenset(grupos): MappingProxyType(dict(zip(_GANADORES_CON_TERCERO, rivales)))
  for grupos, rivales in {
    "ABCD": "CDAB", "ABCE": "CABE", "ABCF": "CABF", "ABDE": "DABE", "ABDF": "DABF",
    "ABEF": "EABF", "ACDE": "CDAE", "ACDF": "CDAF", "ACEF": "CAFE", "ADEF": "DAFE",
    "BCDE": "CDBE", "BCDF": "CDBF", "BCEF": "ECBF", "BDEF": "EDBF", "CDEF": "CDFE",
  }.items()
})

# Orden de juego del 1er round de la fase final. Notacion "<posicion><grupo>": "2E" = Runners-up Grupo E.
# "1A-3" = el ganador del grupo A enfrenta al mejor tercero que le asignen los cruces
_LLAVES_1986_1994 = MappingProxyType({
  1986: ("1A-3", "1F-2E", "2A-2C", "1B-3", "1C-3", "2F-2B", "1E-2D", "1D-3"),
  1990: ("1A-3", "1E-2D", "2F-2B", "1B-3", "2A-2C", "1C-3", "1D-3", "1F-2E"),
  1994: ("2A-2C", "1A-3", "1B-3", "1F-2E", "1C-3", "1E-2D", "1D-3", "2F-2B"),
})

def _etiqueta(codigo):
  """Traduce un codigo de llave ("2E") a la etiqueta que entiende "camb_grp_elm_smpl" ("Runners-up Grupo E")"""
  return f"{_POSICIONES[codigo[0]]} Grupo {codigo[1]}"

class ReglamentoInvalido(ValueError):
  """Error con la lista completa de problemas del reglamento, para mostrarlos todos de una vez en el frontend"""
  def __init__(self, errores):
    self.errores = list(errores)
    super().__init__("Reglamento de mejores terceros invalido:\n- " + "\n- ".join(self.errores))

@dataclass(frozen=True)
class ReglamentoTerceros:
  """
  ¿QUE HACE?
  Representa el reglamento de mejores terceros de un mundial: el orden de los 8 partidos del 1er round de la fase final (llave) y, para cada combinacion de 4 grupos con mejor tercero, el rival de 1A, 1B, 1C y 1D (cruces). Es inmutable y se valida al crearse, sin importar si viene del reglamento FIFA o del frontend

  ¿COMO LO HACE?
  0-) "__post_init__" junta todos los errores de la llave y de los cruces, y si hay alguno lanza (ReglamentoInvalido)

  1-) La llave es valida si tiene 8 partidos con formato "1A-2C" / "1A-3", cada 1ro y 2do de grupo aparece exactamente una vez, solo 1A, 1B, 1C y 1D enfrentan a un mejor tercero y no hay cruces entre equipos del mismo grupo

  2-) Los cruces son validos si estan las 15 combinaciones, cada combinacion reparte sus 4 grupos entre 1A, 1B, 1C y 1D, y ningun ganador enfrenta al tercero de su propio grupo

  Atributos:

    - agno: año del mundial (1986, 1990 o 1994)

    - llave: tupla de 8 partidos en notacion "<posicion><grupo>-<posicion><grupo>"

    - cruces: mapeo frozenset(4 grupos) -> {"A": grupo del 3ro rival, "B": ..., "C": ..., "D": ...}
  """
  agno: int
  llave: tuple
  cruces: MappingProxyType

  def __post_init__(self):
    errores = [*self._errores_llave(), *self._errores_cruces()]
    if self.agno not in _LLAVES_1986_1994:
      errores.insert(0, f"el mundial {self.agno} no usa el formato de mejores terceros (solo 1986, 1990 y 1994)")
    if errores:
      raise ReglamentoInvalido(errores)

  def _errores_llave(self):
    if len(self.llave) != 8:
      yield f"la llave debe tener 8 partidos y tiene {len(self.llave)}"
    malos = [p for p in self.llave if not (isinstance(p, str) and _PATRON_PARTIDO.fullmatch(p))]
    if malos:
      yield f"partidos con formato invalido: {malos}"
      return
    cruces = [p.split("-") for p in self.llave]
    fijos = [c for cruce in cruces for c in cruce if c != "3"]
    if repetidos := sorted({c for c in fijos if fijos.count(c) > 1}):
      yield f"equipos repetidos en la llave: {repetidos}"
    if faltan := sorted(_CLASIFICADOS_FIJOS - set(fijos)):
      yield f"equipos sin partido en la llave: {faltan}"
    contra_tercero = {local for local, visita in cruces if visita == "3"}
    if contra_tercero != {"1" + g for g in _GANADORES_CON_TERCERO}:
      yield f"solo 1A, 1B, 1C y 1D enfrentan a un mejor tercero (recibido: {sorted(contra_tercero)})"
    if mismos := [f"{l}-{v}" for l, v in cruces if v != "3" and l[1] == v[1]]:
      yield f"cruces entre equipos del mismo grupo: {mismos}"

  def _errores_cruces(self):
    if set(self.cruces) != _COMBINACIONES:
      yield "deben definirse los cruces para las 15 combinaciones de 4 grupos"
      return
    for grupos, rival_de in self.cruces.items():
      nombre = "".join(sorted(grupos))
      if sorted(rival_de) != list(_GANADORES_CON_TERCERO) or sorted(rival_de.values()) != sorted(grupos):
        yield f"{nombre}: cada grupo debe aportar exactamente un rival a 1A, 1B, 1C y 1D"
      elif mismos := [g for g, rival in rival_de.items() if g == rival]:
        yield f"{nombre}: el ganador de {mismos} no puede enfrentar al tercero de su propio grupo"

  @classmethod
  def oficial(cls, agno):
    """Reglamento FIFA del año (agno)"""
    return cls(agno, _LLAVES_1986_1994.get(agno, ()), _CRUCES_TERCEROS)

  @classmethod
  def desde_json(cls, agno, datos):
    """Construye (y valida) el reglamento enviado por el frontend: {"llave": [...8], "cruces": {"ABCD": "CDAB", ...}}"""
    try:
      llave = tuple(datos["llave"])
      cruces = {frozenset(grupos): MappingProxyType(dict(zip(_GANADORES_CON_TERCERO, rivales)))
                for grupos, rivales in datos["cruces"].items()}
    except (KeyError, TypeError, AttributeError):
      raise ReglamentoInvalido(['se esperaba {"llave": [8 partidos], "cruces": {"ABCD": "CDAB", ...}}']) from None
    return cls(agno, llave, MappingProxyType(cruces))

  def a_json(self):
    """Inverso de "desde_json", para precargar el editor del frontend"""
    return {"llave": list(self.llave),
            "cruces": {"".join(sorted(grupos)): "".join(rival_de[g] for g in _GANADORES_CON_TERCERO)
                       for grupos, rival_de in sorted(self.cruces.items(), key=lambda kv: sorted(kv[0]))}}

def reglamentos_oficiales():
  """Diccionario {agno: reglamento FIFA en formato JSON} para entregar al template"""
  return {agno: ReglamentoTerceros.oficial(agno).a_json() for agno in _LLAVES_1986_1994}

# --- tabla_terceros(dict_table, df_grupos=None): IGUAL QUE EN LA RESPUESTA ANTERIOR ---

def con_reglamento_terceros(func):
  """
  ¿QUE HACE?
  Decorador que entrega a la funcion decorada el reglamento de mejores terceros a usar: el personalizado si quien llama lo pasa en (personalizado), o el oficial FIFA de (agno) si no. La funcion decorada solo alinea equipos con las tablas de grupos ya jugadas, sin saber de donde salio el reglamento
  """
  @wraps(func)
  def envoltura(dict_table, agno, *args, personalizado=None, **kwargs):
    reglamento = personalizado if personalizado is not None else ReglamentoTerceros.oficial(agno)
    if reglamento.agno != agno:
      raise ValueError(f"el reglamento es del mundial {reglamento.agno} y se pidio el {agno}")
    return func(dict_table, reglamento, *args, **kwargs)
  return envoltura

@con_reglamento_terceros
def alinear_knockout_1986_1994(dict_table, reglamento, df_grupos=None):
  """
  ¿QUE HACE?
  Genera el df (home, away) del 1er round de la fase final segun los 4 mejores terceros de las tablas de grupos ya jugadas y el (reglamento) que inyecta "con_reglamento_terceros". Quien llama pasa (dict_table, agno, df_grupos=..., personalizado=...)

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
