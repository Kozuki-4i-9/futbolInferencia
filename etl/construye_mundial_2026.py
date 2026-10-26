"""
Mundial 2026 (48 equipos, 12 grupos, 8 mejores terceros) en las tablas de la app: grupos, fixture, partidos y el
reglamento oficial de mejores terceros.

Fuentes:
  - data/partidos/internacionales.csv (etl/busca_internacionales.py): los 104 partidos con fecha y marcador
  - Wikipedia en ingles: "2026 FIFA World Cup" (tablas de los 12 grupos, con el orden oficial de cada grupo) y
    "2026 FIFA World Cup knockout stage" (cruces de dieciseisavos, la tabla de las 495 combinaciones de mejores terceros y
    las tandas de penales)

Genera / actualiza:
  - data/partidos/terceros_2026.csv: combinacion de grupos con mejor tercero -> rival de 1A, 1B, 1D, 1E, 1G, 1I, 1K, 1L
  - data/grupos/grupos_mundiales.csv, data/partidos/fixtures.csv y data/partidos/partidos.csv: las filas de 2026
    (se reemplazan si ya estaban)
  - data/partidos/clasificaciones.csv: las eliminatorias desde ELIMINATORIAS_DESDE (estaban incompletas: 156 de 255
    partidos de 2024 y ninguno de 2025 ni 2026), tomadas de internacionales.csv en orden de fecha

El cuadro de octavos en adelante ("Winner Match 74 vs Winner Match 77") se deduce de los resultados: cada partido
enfrenta a los ganadores de dos partidos anteriores. Antes de escribir se valida que el reglamento oficial, aplicado a
los grupos reales, da exactamente los dieciseisavos que se jugaron.

Uso (descarga las dos paginas):
    python etl/construye_mundial_2026.py
    python etl/construye_mundial_2026.py --html-principal p.html --html-eliminatorias e.html   # paginas ya guardadas
"""
import argparse
import re
import sys
import urllib.request
from itertools import combinations

import pandas as pd
from bs4 import BeautifulSoup

from rutas import DATA, PARTIDOS

AGNO = 2026
URL = "https://en.wikipedia.org/wiki/"
PAGINAS = {"principal": "2026_FIFA_World_Cup", "eliminatorias": "2026_FIFA_World_Cup_knockout_stage"}
GRUPOS = "ABCDEFGHIJKL"
GANADORES_CON_TERCERO = "ABDEGIKL"
PRIMER_PARTIDO_ELIMINATORIA = 73
ELIMINATORIAS_DESDE = 2024

# nombres de Wikipedia que no estan en la traduccion de martj42 (data/partidos/nombres_en_es.csv)
ALIAS = {"DR Congo": "DR Congo", "Ivory Coast": "Ivory Coast", "Curaçao": "Curaçao", "Türkiye": "Turkey"}
POSICION = {"Winner": "1", "Runner-up": "2", "Third-place": "3"}


def descargar(pagina):
  pedido = urllib.request.Request(URL + pagina, headers={"User-Agent": "futbolInferencia/1.0 (proyecto personal)"})
  return urllib.request.urlopen(pedido, timeout=60).read().decode("utf-8")


def texto(nodo):
  return " ".join(nodo.get_text(" ").split())


def leer_grupos(html, traducir):
  """
  Las 12 tablas "Group X" de la pagina principal, en el orden oficial de cada grupo

  Returns:

    - df con Grupo ("Grupo A"), posicion, pais, Pts, PJ, PG, PE, PP, GF, GC, Dif
  """
  filas = []
  for tabla in BeautifulSoup(html, "html.parser").find_all("table"):
    trs = tabla.find_all("tr")
    cabecera = [c.get_text(strip=True) for c in trs[0].find_all(["th", "td"])] if trs else []
    titulo = tabla.find_previous(["h3", "h2"])
    grupo = re.fullmatch(r"Group ([A-L])", titulo.get_text(strip=True)) if titulo else None
    if cabecera[:1] != ["Pos"] or "Pts" not in cabecera or not grupo:
      continue
    for tr in trs[1:]:
      c = [re.sub(r"\[[^\]]*\]", "", x.get_text(strip=True)) for x in tr.find_all(["th", "td"])]   # sin notas al pie "[a]"
      equipo = re.sub(r"\((H|X|Y|E)\)$", "", c[1]).strip()
      filas.append({"Grupo": f"Grupo {grupo.group(1)}", "posicion": int(c[0]), "pais": traducir(equipo),
                    "PJ": int(c[2]), "PG": int(c[3]), "PE": int(c[4]), "PP": int(c[5]), "GF": int(c[6]), "GC": int(c[7]),
                    "Dif": c[8].replace("−", "-").replace("+", ""), "Pts": int(c[9])})
  grupos = pd.DataFrame(filas).drop_duplicates(["Grupo", "pais"])
  if sorted(grupos["Grupo"].unique()) != [f"Grupo {g}" for g in GRUPOS] or len(grupos) != 48:
    raise SystemExit(f"Se esperaban 12 grupos de 4 y se leyeron {grupos.groupby('Grupo').size().to_dict()}")
  return grupos.sort_values(["Grupo", "posicion"], ignore_index=True)


def leer_eliminatorias(html):
  """
  ¿QUE HACE?
  De la pagina de la fase final: los cruces de dieciseisavos (Match 73 a 88), la tabla de las 495 combinaciones de mejores terceros y las tandas de penales

  Returns:

    - (llave, cruces, tandas): llave = lista de 16 codigos ("2A-2B", "1E-3"); cruces = {"EFGHIJKL": "EJIFHGLK", ...} (rival de cada ganador de GANADORES_CON_TERCERO); tandas = {(fecha, home, away en ingles): "a:b"}
  """
  sopa = BeautifulSoup(html, "html.parser")
  plano = texto(sopa)
  llave = []
  for n in range(PRIMER_PARTIDO_ELIMINATORIA, PRIMER_PARTIDO_ELIMINATORIA + 16):
    m = re.search(rf"Match {n} : (Winner|Runner-up) Group ([A-L]) vs (Winner|Runner-up|Third-place) Group ([A-L/]+)", plano)
    if not m:
      raise SystemExit(f"No se encontro el cruce del Match {n}")
    local = POSICION[m.group(1)] + m.group(2)
    visita = "3" if m.group(3) == "Third-place" else POSICION[m.group(3)] + m.group(4)
    llave.append(f"{local}-{visita}")

  cruces = {}
  for tabla in sopa.find_all("table"):
    trs = tabla.find_all("tr")
    if len(trs) < 400:
      continue
    for tr in trs[1:]:
      c = [x.get_text(strip=True) for x in tr.find_all(["th", "td"])]
      if not c or not re.match(r"\d+\*?$", c[0]):   # "67*" marca la combinacion que se dio en 2026
        continue
      grupos = "".join(x for x in c[1:13] if re.fullmatch(r"[A-L]", x))
      rivales = "".join(x[1] for x in c[-8:])
      cruces[grupos] = rivales
  if len(cruces) != 495:
    raise SystemExit(f"La tabla de combinaciones tiene {len(cruces)} filas y se esperaban 495")

  tandas = {}
  for caja in sopa.select("div.footballbox"):
    t = texto(caja)
    m = re.search(r"\( (\d{4}-\d{2}-\d{2}) \).*? UTC\S* (?:\[ \w \] )?(.+?) (\d+)–(\d+) \( a\.e\.t\. \) (.+?) (?:[A-ZÁ-Úa-zá-úø'\. -]+ \d|\[ Report)", t)
    pen = re.search(r"Penalties .*? (\d+)–(\d+) ", t)
    if m and pen:
      tandas[(m.group(1), m.group(2).strip(), m.group(5).strip())] = f"{pen.group(1)}:{pen.group(2)}"
  return llave, cruces, tandas


def main():
  sys.stdout.reconfigure(encoding="utf-8")
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--html-principal")
  parser.add_argument("--html-eliminatorias")
  args = parser.parse_args()
  leer = lambda ruta, clave: open(ruta, encoding="utf-8").read() if ruta else descargar(PAGINAS[clave])

  nombres = dict(pd.read_csv(PARTIDOS / "nombres_en_es.csv", encoding="utf-8").itertuples(index=False))
  internacionales = pd.read_csv(PARTIDOS / "internacionales.csv", encoding="utf-8")
  mundial = internacionales[(internacionales["categoria"] == "mundial") & (internacionales["agno"] == AGNO)].sort_values("fecha", kind="stable")
  equipos = set(mundial["home"]) | set(mundial["away"])
  # nombre de la BD -> nombre en ingles de martj42 (para las tandas, que vienen en ingles)
  a_ingles = {es: en for en, es in nombres.items()}

  def traducir(nombre):
    for candidato in (nombres.get(nombre), nombres.get(ALIAS.get(nombre, "")), nombre):
      if candidato in equipos:
        return candidato
    raise SystemExit(f"No se pudo traducir '{nombre}' a un equipo del mundial 2026 de internacionales.csv")

  grupos = leer_grupos(leer(args.html_principal, "principal"), traducir)
  llave, cruces, tandas_en = leer_eliminatorias(leer(args.html_eliminatorias, "eliminatorias"))
  grupo_de = dict(zip(grupos["pais"], grupos["Grupo"].str[-1]))
  codigo = {f"{p}{g[-1]}": pais for g, p, pais in zip(grupos["Grupo"], grupos["posicion"], grupos["pais"])}

  # partidos de grupo (los dos del mismo grupo) y de eliminatoria
  de_grupo = mundial.apply(lambda r: grupo_de.get(r.home) == grupo_de.get(r.away), axis=1)
  fase_grupos, eliminatoria = mundial[de_grupo], mundial[~de_grupo]
  if len(fase_grupos) != 72 or len(eliminatoria) != 32:
    raise SystemExit(f"Se esperaban 72 partidos de grupo y 32 de eliminatoria: {len(fase_grupos)} y {len(eliminatoria)}")

  # tandas: de los nombres de Wikipedia (ingles) a la orientacion de internacionales
  tandas = {}
  for (fecha, h_en, a_en), tanda in tandas_en.items():
    h, a = traducir(h_en), traducir(a_en)
    if ((eliminatoria["fecha"] == fecha) & (eliminatoria["home"] == h) & (eliminatoria["away"] == a)).any():
      tandas[(fecha, h, a)] = tanda
    else:
      p0, p1 = tanda.split(":")
      tandas[(fecha, a, h)] = f"{p1}:{p0}"

  def ganador(r):
    if r.score_0 != r.score_1:
      return (r.home, r.away) if r.score_0 > r.score_1 else (r.away, r.home)
    p0, p1 = (int(x) for x in tandas[(r.fecha, r.home, r.away)].split(":"))
    return (r.home, r.away) if p0 > p1 else (r.away, r.home)

  # 1) dieciseisavos: cada partido real se ubica por el equipo fijo de su cruce (1E, 2A, ...)
  terceros = grupos[grupos["posicion"] == 3]
  primeros_partidos = list(eliminatoria.itertuples(index=False))[:16]
  numerado, usados = {}, set()
  for n, cruce in enumerate(llave, start=PRIMER_PARTIDO_ELIMINATORIA):
    fijo = codigo[cruce.split("-")[0]]
    partido = next(r for r in primeros_partidos if fijo in (r.home, r.away))
    numerado[n] = partido
    usados.add(partido)
  if len(usados) != 16:
    raise SystemExit("Los dieciseisavos reales no encajan con los cruces leidos")

  # validacion: el reglamento oficial, con los 8 terceros que pasaron, da los rivales que se jugaron
  pasaron = "".join(sorted(grupo_de[t] for r in primeros_partidos for t in (r.home, r.away) if t in set(terceros["pais"])))
  rival_de = dict(zip(GANADORES_CON_TERCERO, cruces[pasaron]))
  for n, cruce in enumerate(llave, start=PRIMER_PARTIDO_ELIMINATORIA):
    if cruce.endswith("-3"):
      esperado = {codigo[cruce[:2]], codigo["3" + rival_de[cruce[1]]]}
      real = {numerado[n].home, numerado[n].away}
      if esperado != real:
        raise SystemExit(f"Match {n}: el reglamento da {esperado} y se jugo {real}")
  print(f"Terceros que pasaron: {pasaron}; el reglamento oficial reproduce los 16 dieciseisavos")

  # 2) octavos en adelante: cada partido enfrenta a los ganadores (o perdedores, en el 3er puesto) de dos anteriores
  resto = list(eliminatoria.itertuples(index=False))[16:]
  referencia = {}
  for n, r in numerado.items():
    g, p = ganador(r)
    referencia[g] = f"Winner Match {n}"
  filas_cuadro, perdedor_ref = [], {}
  for n, r in enumerate(resto, start=PRIMER_PARTIDO_ELIMINATORIA + 16):
    if n == PRIMER_PARTIDO_ELIMINATORIA + 30:   # 3er puesto: perdedores de las semifinales
      ref_h, ref_a = perdedor_ref[r.home], perdedor_ref[r.away]
    else:
      ref_h, ref_a = referencia[r.home], referencia[r.away]
    filas_cuadro.append((ref_h, f"Match {n}", ref_a))
    numerado[n] = r
    g, p = ganador(r)
    referencia[g] = f"Winner Match {n}"
    if n in (PRIMER_PARTIDO_ELIMINATORIA + 28, PRIMER_PARTIDO_ELIMINATORIA + 29):   # semifinales
      perdedor_ref[p] = f"Loser Match {n}"

  # 3) salidas
  pd.DataFrame(sorted(cruces.items()), columns=["combinacion", "rivales"]).to_csv(PARTIDOS / "terceros_2026.csv", index=False, encoding="utf-8")

  etiqueta = {"1": "Winner Grupo", "2": "Runners-up Grupo"}
  orden_grupo = fase_grupos.assign(grupo=fase_grupos["home"].map(grupo_de)).sort_values(["grupo", "fecha"], kind="stable")
  partidos_2026 = [(r.home, f"{r.score_0}:{r.score_1}", r.away, AGNO, "", 1) for r in orden_grupo.itertuples(index=False)]
  partidos_2026 += [(r.home, f"{r.score_0}:{r.score_1}", r.away, AGNO, tandas.get((r.fecha, r.home, r.away), ""), 1)
                    for n, r in sorted(numerado.items())]
  fixture_2026 = [(r.home, f"{r.score_0}:{r.score_1}", r.away, AGNO) for r in orden_grupo.itertuples(index=False)]
  for n, cruce in enumerate(llave, start=PRIMER_PARTIDO_ELIMINATORIA):
    local, visita = cruce.split("-")
    if visita == "3":   # el tercero real de ese cruce (la app lo vuelve a asignar con el reglamento al simular)
      tercero = next(t for t in (numerado[n].home, numerado[n].away) if t != codigo[local])
      visita_etiqueta = f"Third-p Grupo {grupo_de[tercero]}"
    else:
      visita_etiqueta = f"{etiqueta[visita[0]]} {visita[1]}"
    fixture_2026.append((f"{etiqueta[local[0]]} {local[1]}", f"Match {n}", visita_etiqueta, AGNO))
  fixture_2026 += [(h, s, a, AGNO) for h, s, a in filas_cuadro]

  def reemplazar(ruta, filas, columnas, col_agno, con_id):
    df = pd.read_csv(ruta, dtype=str, keep_default_na=False, encoding="utf-8")
    df = df[df[col_agno] != str(AGNO)]
    nuevo = pd.DataFrame(filas, columns=columnas).astype(str)
    salida = pd.concat([df.drop(columns=["id"]) if con_id else df, nuevo], ignore_index=True)
    if con_id:
      salida.insert(0, "id", range(len(salida)))
    salida.to_csv(ruta, index=False, encoding="utf-8")

  reemplazar(PARTIDOS / "partidos.csv", partidos_2026, ["home", "score", "away", "year", "penales", "jugado"], "year", con_id=False)
  reemplazar(PARTIDOS / "fixtures.csv", fixture_2026, ["home", "score", "away", "agno"], "agno", con_id=True)
  eliminatorias = internacionales[(internacionales["categoria"] == "eliminatoria_mundial") & (internacionales["agno"] >= ELIMINATORIAS_DESDE)]
  eliminatorias = eliminatorias.sort_values("fecha", kind="stable")
  clasif = pd.read_csv(PARTIDOS / "clasificaciones.csv", dtype=str, keep_default_na=False, encoding="utf-8")
  clasif = clasif[clasif["agno"].astype(int) < ELIMINATORIAS_DESDE].drop(columns=["id"])
  nuevas = eliminatorias[["agno", "home", "away", "score_0", "score_1", "tournament"]].astype(str)
  clasif = pd.concat([clasif, nuevas], ignore_index=True)
  clasif.insert(0, "id", range(len(clasif)))
  clasif.to_csv(PARTIDOS / "clasificaciones.csv", index=False, encoding="utf-8")
  print(f"clasificaciones: {len(nuevas)} partidos de eliminatorias desde {ELIMINATORIAS_DESDE}")

  grupos_salida = grupos.assign(agno=AGNO)[["pais", "Pts", "PJ", "PG", "PE", "PP", "GF", "GC", "Dif", "Grupo", "agno"]]
  reemplazar(DATA / "grupos" / "grupos_mundiales.csv", grupos_salida.values.tolist(), list(grupos_salida.columns), "agno", con_id=True)
  print(f"2026: {len(partidos_2026)} partidos ({len(tandas)} definidos por penales), {len(fixture_2026)} filas de fixture, "
        f"{len(grupos)} filas de grupos, {len(cruces)} combinaciones de mejores terceros")


if __name__ == "__main__":
  main()
