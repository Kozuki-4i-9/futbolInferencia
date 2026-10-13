# futbolInferencia

Predicción de resultados de los Mundiales de fútbol (1934 – 2022) con un modelo de **series temporales (LSTM)**. Una aplicación web en Django deja al usuario elegir un mundial y una fase, modificar los partidos que quiera y simular el resto del torneo partido a partido. El modelo recibe el rendimiento acumulado de cada selección y lo actualiza tras cada predicción.

---

## Objetivos

### Objetivo general
Construir un sistema que **simule un Mundial completo** a partir de cualquier fase elegida por el usuario: predice los goles de cada partido con una red LSTM y propaga el rendimiento de cada selección de una fase a la siguiente.

### Objetivos específicos
1. **Datos:** reunir partidos, fixtures, grupos y eliminatorias de todos los mundiales en PostgreSQL (scraping en `scripts/activos/`).
2. **Rendimiento base:** acumular por selección, partido a partido, las métricas `PTS PJ PG PP PE GF GC D` desde las eliminatorias previas hasta la fase en curso.
3. **Rendimiento derivado:** calcular métricas opcionales (`gkps mds mos mms rate_GC rate_GF`) a partir de insumos abstractos (`VI SOT PKATT PKATTALLOW`) que también avanzan partido a partido.
4. **Inferencia secuencial:** alimentar la LSTM con ventanas temporales de partidos (embedding del equipo + métricas) y convertir los goles predichos en resultados mediante la intersección de probabilidades de Poisson.
5. **Formatos de torneo:** respetar cada formato histórico (eliminación directa en 1934/38, fase final de grupos en 1950, mejores terceros en 1986 – 1994, octavos desde 1998).
6. **Control del usuario:** dejar elegir desde el frontend qué métricas base descartar y qué métricas opcionales usar, **sin cambiar las dimensiones** de la entrada al modelo.
7. **Escalabilidad:** preparar la lógica de fases para otros torneos (Champions League, Copa Libertadores).

---

## Cómo funciona la IA de series temporales

```mermaid
flowchart TD
    subgraph DATOS["🗄️ Datos (PostgreSQL)"]
        A1[(partidos)]
        A2[(clasificaciones)]
        A3[(fixtures / grupos)]
        A4[(proxy_desempegno<br/>VI · SOT · PKATT · PKATTALLOW)]
    end

    subgraph ACUM["📈 Rendimiento acumulado (fase a fase)"]
        B1["funcion_tabla_desempegno<br/>BASE: PTS PJ PG PP PE GF GC D"]
        B2["opcionales (progreso por país)<br/>insumos abstractos"]
    end

    subgraph FEAT["🧮 Entrada al modelo — calculo_metricas_0"]
        C1["Base<br/>PTS PJ PG PP PE GF GC D"]
        C2["Derivadas<br/>gkps mds mos mms rate_GC rate_GF"]
        C3{"🎛️ Elección del usuario<br/>drop_metricas / metricas_opcionales"}
        C4["Esquema fijo: 14 métricas por equipo<br/>lo apagado ⇒ VALOR_NEUTRO"]
    end

    subgraph MODELO["🧠 Modelo"]
        D1["Embedding del equipo (3D)"]
        D2["Ventanas temporales — c_s<br/>(N_STEPS partidos por muestra)"]
        D3["LSTM 100 → Dropout 0.20<br/>LSTM 57 → Dropout 0.40<br/>LSTM 25 → Dropout 0.15<br/>Dense 32 ReLU → Dense 1"]
        D4["Goles esperados<br/>home · away"]
    end

    subgraph RES["⚽ Resultado"]
        E1["Intersección de Poisson<br/>probabilidad del marcador"]
        E2["Puntos de grupo / ganador de llave"]
    end

    A2 --> B1
    A1 --> B1
    A3 --> B1
    A4 --> B2
    B1 --> C1
    B1 --> C2
    B2 --> C2
    C1 --> C3
    C2 --> C3
    C3 --> C4
    C4 --> D1 --> D2 --> D3 --> D4
    D4 --> E1 --> E2
    E2 -- "calculo_metricas_1:<br/>actualiza base y progreso" --> B1
    E2 -. "siguiente fase" .-> B2

    classDef datos fill:#e8f1fb,stroke:#3b82c4,color:#0b2540
    classDef acum fill:#eaf7ee,stroke:#2f9e57,color:#0d3320
    classDef feat fill:#fff6e5,stroke:#d38b00,color:#3d2800
    classDef modelo fill:#f3ecfb,stroke:#7c4dbd,color:#2a1446
    classDef res fill:#fdecec,stroke:#c94848,color:#3f1010
    class A1,A2,A3,A4 datos
    class B1,B2 acum
    class C1,C2,C3,C4 feat
    class D1,D2,D3,D4 modelo
    class E1,E2 res
```

**La idea central:** cada partido simulado modifica el rendimiento de las dos selecciones, y ese rendimiento es la serie temporal que ve el modelo en el partido siguiente. Así el torneo se predice como una secuencia, no como partidos aislados.

### Neutralización de métricas (control del usuario)

```mermaid
flowchart LR
    U["Frontend<br/>casillas de métricas"] -->|"drop_metricas<br/>metricas_opcionales"| V["views.py<br/>valida contra el catálogo"]
    V -->|"400 si hay nombres desconocidos<br/>o falta un valor neutro"| U
    V --> M["consulta_general → func_prediccion_orden"]
    M --> N["calculo_metricas_0 + neutralizar"]
    N --> X["Entrada al modelo<br/>siempre 2 × 16 columnas"]
```

- **Base descartada** (`drop_metricas`): su columna llega al modelo con `VALOR_NEUTRO`, pero se sigue acumulando en la simulación.
- **Opcional no aceptada** (no figura en `metricas_opcionales`): su columna llega con `VALOR_NEUTRO`.
- Ninguna columna se quita ni se añade, así que el modelo recibe siempre las mismas dimensiones.

---

## Estructura del proyecto

```
futbolInferencia/
├── data/                     # datos y modelos entrenados (fuera de git)
├── scripts/
│   ├── activos/              # scraping, limpieza y notebook de entrenamiento
│   └── pruebas/              # prototipos y notas de avance (avances.py)
└── web/                      # aplicación Django
    ├── mundial/              # settings (lee .env con python-decouple)
    └── fases/
        ├── models.py         # partidos, fixtures, grupos, clasificaciones, proxy_desempegno
        ├── modulo.py         # métricas, features, inferencia y reglamentos de mejores terceros
        ├── views.py          # endpoints: procesar_recepcion_0 (página) y _1 (predecir)
        └── templates/prediccion_2.html
```

## Puesta en marcha

1. Copiar `web/.env.example` a `web/.env` y completar `SECRET_KEY`, `DEBUG`, `ALLOWED_HOSTS`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`, `DB_HOST` y `DB_PORT`.
2. `cd web && python manage.py migrate` (crea también la tabla `proxy_desempegno`).
3. Cargar los datos en las tablas, incluida `proxy_desempegno` con su columna `pais`.
4. `python manage.py runserver`.

> `modulo.py` consulta la base de datos al importarse (arma `dic_agnos`), así que Django necesita la BD disponible incluso para `makemigrations`.

**Stack:** Python · Django · PostgreSQL · TensorFlow/Keras · pandas · NumPy · SciPy · JavaScript (fetch).

---

## Historia del proyecto

```mermaid
timeline
    title futbolInferencia 2026
    Abril : Commit inicial y app web : Módulo de inferencia (modulo_8) : Primera corrección de docstrings
    Agosto : Reorganización y planes de optimización : calculo_metricas_0 optimizado (modulo_10 / modulo_11) : calc_gkps, calc_mds, calc_mos, calc_mms : Tabla proxy_desempegno : Partidos default antes de la fase a predecir
    Septiembre : Ajuste de calculo_metricas_0 en notebooks : Insumos para métricas opcionales : Nueva fase_de_grupos (base para Champions League)
    Octubre : Seguridad de arranque (.env) : Progreso partido a partido de métricas abstractas : Reglamento de mejores terceros 1986–1994 : Esquema fijo y neutralización de métricas
```

| Fecha | Cambio |
|---|---|
| 2026-04-12 | Commit inicial, primera versión y app web. |
| 2026-04-13 / 19 | Ajustes a `modulo_8.py` y primera ronda de docstrings. |
| 2026-08-10 / 11 | Estructura y planes de optimización; se retira data innecesaria. |
| 2026-08-15 | `modulo_10.py`: `calculo_metricas_0` pensado para el máximo de métricas de rendimiento. |
| 2026-08-29 | Funciones `calc_gkps`, `calc_mds`, `calc_mos` y `calc_mms`; modelo `proxy_desempegno`. |
| 2026-08-30 | `fase_de_grupos` y `jugar` aceptan resultados default antes de la fase a predecir; `modulo_11.py`. |
| 2026-09-02 / 05 | Notebooks para ajustar `calculo_metricas_0`; propuestas para `funcion_tabla_desempegno`. |
| 2026-09-27 / 10-01 | Insumos para métricas opcionales; refactor para escalabilidad; nueva `fase_de_grupos`. |
| 2026-10-07 | Rama `fix/seguridad-arranque`: settings con `.env`; se dejan de versionar datos y notebooks. |
| 2026-10-11 | `funcion_tabla_desempegno` y `calculo_metricas_1` avanzan el progreso de las abstractas partido a partido. |
| 2026-10-12 | Reglamento de mejores terceros 1986 – 1994 (oficial o personalizado desde el frontend). |
| 2026-10-12 | **Esquema fijo de entrada y neutralización de métricas**: `BASE_METRICS` + `OPTIONAL_METRICS` = 14 por equipo; descarte y elección de métricas desde el frontend; histórico abstracto por id de país; métricas derivadas calculadas por partido; migración de `proxy_desempegno`. |

---

## Pendientes conocidos

- **Reentrenar el modelo** con el esquema de `ALL_METRICS` (14 métricas por equipo, mismo orden y misma `calculo_metricas_0`). El `.h5` actual (diciembre de 2024) se entrenó con 10 features por paso.
- **`VALOR_NEUTRO`:** rellenarlo con la media de cada columna de `X_tr`. Hasta entonces el servidor rechaza (400) cualquier intento de descartar métricas o de no usar alguna opcional.
- **PJ histórico en `proxy_desempegno`:** VI, SOT y PKATT son totales históricos, y sin su PJ los promedios se calculan como si fueran un solo partido.
- **`N_STEPS`:** vale 0 en `modulo.py`, pero el entrenamiento usó ventanas de 5 partidos.
- **Estado global:** `opcionales` se reinicia en cada consulta, pero dos consultas simultáneas en un servidor con varios hilos lo compartirían. Habría que moverlo dentro del objeto de predicción.
- **Ruta del modelo** fija en `consulta_general`: llevarla a `.env`.
- Adaptación a Champions League y Copa Libertadores.
