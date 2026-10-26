"""
Compara lo redescargado en data/redescarga/ con los datos que usa hoy el proyecto y escribe el resultado en
data/redescarga/informe.md:

  1. Grupos (Wikipedia) contra data/grupos/grupos_mundiales_<año>.csv: selecciones, grupo y los 8 numeros de la tabla
  2. Partidos de mundiales (Wikipedia) contra data/partidos/partidos_<año>.csv: cantidad, equipos y marcador, y que
     partidos definidos por penales quedaron guardados con el marcador de la tanda
  3. Eliminatorias (martj42) contra data/partidos/clasificaciones.csv: cantidad de partidos y marcadores por año

Uso (tras descargar con etl/busca_grupos.py --todos --salida data/redescarga/grupos,
busca_partidos.py --todos y descarga_internacionales.py):
    python experimentos/redescarga/comparar.py
"""
import re
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
ACTUAL = RAIZ / "data"
NUEVO = RAIZ / "data" / "redescarga"


def numero(valor):
    """'+3', '−2', '3' -> float; lo demas se devuelve tal cual"""
    v = str(valor).replace("−", "-").replace("–", "-").replace("+", "").strip()
    return float(v) if re.fullmatch(r"-?\d+(\.\d+)?", v) else v


def comparar_grupos():
    lineas = ["## 1. Grupos (Wikipedia)", "", "| Año | Filas actual / nuevo | Resultado |", "|---|---|---|"]
    for ruta in sorted((NUEVO / "grupos").glob("grupos_mundiales_*.csv")):
        anio = ruta.stem.rsplit("_", 1)[-1]
        nuevo = pd.read_csv(ruta, index_col=0)
        actual = pd.read_csv(ACTUAL / "grupos" / ruta.name, index_col=0)
        problemas = []
        faltan, sobran = sorted(set(actual.index) - set(nuevo.index)), sorted(set(nuevo.index) - set(actual.index))
        if faltan or sobran:
            problemas.append(f"solo en actual {faltan}, solo en nuevo {sobran}")
        comunes = [e for e in actual.index if e in nuevo.index]
        grupos = [e for e in comunes if str(actual.at[e, "Grupo"]).lower() != str(nuevo.at[e, "Grupo"]).lower()]
        if grupos:
            problemas.append(f"grupo distinto: {grupos}")
        a = actual.loc[comunes].iloc[:, :8].map(numero)
        b = nuevo.loc[comunes].iloc[:, :8].map(numero)
        for equipo in comunes:
            if list(a.loc[equipo]) != list(b.loc[equipo]):
                problemas.append(f"{equipo}: actual {list(a.loc[equipo])} / nuevo {list(b.loc[equipo])}")
        lineas.append(f"| {anio} | {len(actual)} / {len(nuevo)} | {'igual' if not problemas else '<br>'.join(problemas)} |")
    return lineas


def comparar_partidos():
    lineas = ["## 2. Partidos de mundiales (Wikipedia)", "",
              "| Año | Partidos actual / nuevo | Penales guardados como marcador | Otras diferencias |", "|---|---|---|---|"]
    for ruta in sorted((NUEVO / "partidos").glob("partidos_*.csv")):
        anio = ruta.stem.rsplit("_", 1)[-1]
        nuevo = pd.read_csv(ruta)
        actual = pd.read_csv(ACTUAL / "partidos" / ruta.name, index_col=0)
        actual.columns = [c.split(";")[0] for c in actual.columns]  # partidos_1998.csv trae "year;;;;"
        actual = actual.assign(year=actual["year"].astype(str).str.extract(r"(\d{4})")[0])
        penales, otras = [], []
        for i in range(max(len(actual), len(nuevo))):
            if i >= len(actual) or i >= len(nuevo):
                fila = nuevo.iloc[i] if i < len(nuevo) else actual.iloc[i]
                otras.append(f"#{i + 1} solo en {'nuevo' if i < len(nuevo) else 'actual'}: {fila['home']} - {fila['away']}")
                continue
            a, n = actual.iloc[i], nuevo.iloc[i]
            marcador_nuevo = f"{n['score_0']}:{n['score_1']}"
            tanda = f"{int(n['penales_0'])}:{int(n['penales_1'])}" if pd.notna(n["penales_0"]) else None
            if (a["home"], a["away"]) != (n["home"], n["away"]):
                otras.append(f"#{i + 1} equipos: actual {a['home']} - {a['away']} / nuevo {n['home']} - {n['away']}")
            elif a["score"] != marcador_nuevo:
                if a["score"] == tanda:
                    penales.append(f"{n['home']} - {n['away']}: {a["score"]} -> {marcador_nuevo} ({tanda} p.)")
                else:
                    otras.append(f"#{i + 1} {n['home']} - {n['away']}: actual {a['score']} / nuevo {marcador_nuevo}")
        lineas.append(f"| {anio} | {len(actual)} / {len(nuevo)} | {'<br>'.join(penales) or '–'} | {'<br>'.join(otras[:6]) or '–'}"
                      f"{f' (+{len(otras) - 6} más)' if len(otras) > 6 else ''} |")
    return lineas


def comparar_eliminatorias():
    actual = pd.read_csv(ACTUAL / "partidos" / "clasificaciones.csv")
    nuevo = pd.read_csv(NUEVO / "internacionales" / "eliminatorias.csv")
    lineas = ["## 3. Eliminatorias (martj42)", "",
              "Los nombres de la fuente estan en ingles: se comparan cantidades y marcadores por año, no equipos.", "",
              "| Año | Partidos actual / nuevo | Marcadores distintos (pares de goles) |", "|---|---|---|"]
    for anio in sorted(set(actual["agno"]) | set(nuevo["agno"])):
        a = actual[actual["agno"] == anio]
        n = nuevo[nuevo["agno"] == anio]
        goles_a = sorted(zip(a["score_0"], a["score_1"]))
        goles_n = sorted(zip(n["home_score"].astype(int), n["away_score"].astype(int)))
        if len(a) == len(n) and goles_a == goles_n:
            continue
        sobrantes = pd.Series(goles_a).value_counts().sub(pd.Series(goles_n).value_counts(), fill_value=0)
        dif = ", ".join(f"{k[0]}:{k[1]} {'+' if v > 0 else ''}{int(v)}" for k, v in sobrantes[sobrantes != 0].items())
        lineas.append(f"| {anio} | {len(a)} / {len(n)} | {dif[:200] or 'mismos marcadores'} |")
    lineas.insert(4, f"Totales: actual {len(actual)} partidos ({actual['agno'].min()} - {actual['agno'].max()}), "
                     f"nuevo {len(nuevo)} ({nuevo['agno'].min()} - {nuevo['agno'].max()}). Solo se listan los años con diferencias.")
    lineas.insert(5, "")
    return lineas


def main():
    informe = ["# Redescarga: comparacion con los datos actuales", ""]
    for seccion in (comparar_grupos, comparar_partidos, comparar_eliminatorias):
        informe += seccion() + [""]
    (NUEVO / "informe.md").write_text("\n".join(informe), encoding="utf-8")
    print("\n".join(informe))


if __name__ == "__main__":
    main()
