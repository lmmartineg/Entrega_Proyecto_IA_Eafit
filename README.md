# Agente de calidad del aire del Valle de Aburrá

Proyecto final de SI3003 Inteligencia Artificial (EAFIT, 2026), idea 3.1 del enunciado.
Este README crece con cada sesión; por ahora cubre la Sesión 1 (datos y EDA).

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate        # Windows (PowerShell): .venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest -q                        # 13 pruebas (datos sintéticos y tabla ICA)
```

En Windows, si `python` abre la Microsoft Store, crea el entorno con la ruta completa de tu Python (por ejemplo `C:\Users\<usuario>\miniconda3\python.exe -m venv .venv`). Una vez activado el entorno, `python` funciona normal.

## Sesión 1: obtener los datos

Todos los comandos se corren desde la raíz del proyecto.

### Fuente principal: Repositorio de Datos SIATA

Dataset: SIATA, 2026, "Material Particulado - PM2.5", [https://doi.org/10.83041/AUWZWT](https://doi.org/10.83041/AUWZWT), Repositorio de Datos SIATA, V5. Un archivo `.tab` por mes. Lee sus condiciones de uso antes de subir los datos al repositorio.

```bash
python scripts/download_siata.py --listar                  # ver qué archivos hay
python scripts/download_siata.py --desde 2019 --hasta 2026 # bajar a data/raw/siata_pm25/
```

Si la API falla (por ejemplo, porque el dataset pide aceptar condiciones), descarga los `.tab` desde la web y ponlos en `data/raw/siata_pm25/`.

Formato verificado el 6 nov 2026: archivos separados por tabuladores, en formato ancho. Tienen una columna `fecha_hora` horaria y una columna por estación (código SIATA, p. ej. `MED-ARAN`) con PM2.5 en µg/m³. Una celda vacía es una hora sin dato. Entre 2019-01 y 2026-09 hay 93 archivos (unos 11 MB) con 30 estaciones.

Luego limpia y agrega a diario:

```bash
python -m src.data process
```

Los archivos de salida están descritos en el paso 4.

### Respaldo: datos.gov.co o CSV manual

Úsalo solo si el SIATA no está disponible. Los pasos 1 a 3 de abajo descargan de datos.gov.co; el 4 procesa ese CSV.

### 1. Ver las columnas reales de la API

```bash
python -m src.data inspect
```

Imprime las columnas del recurso `g4t8-zkc3` de datos.gov.co, 5 filas de ejemplo y un mapeo sugerido. El mapeo es solo una sugerencia por palabras clave: revísalo.

### 2. Configurar `src/config.py`

- Copia los nombres reales de columna en `COLUMNS` (`fecha`, `estacion` y `valor` son obligatorias).
- Pon en `DEPARTAMENTO` y `VARIABLE` los valores exactos que viste en `inspect` para Antioquia y PM2.5.
- Opcional: `MUNICIPIOS` con los municipios del Valle de Aburrá.

### 3. Descargar

```bash
python -m src.data download --desde 2019-01-01
```

Guarda `data/raw/pm25_raw.csv` y `data/raw/pm25_raw.meta.json` (fuente, parámetros, filas y fecha de descarga). Si la API responde 400 por el filtro, repite con `--sin-filtros`; los filtros se aplican luego en la limpieza.

### 4. Limpiar y agregar a diario

```bash
python -m src.data process --raw data/raw/pm25_raw.csv   # respaldo
```

Con cualquiera de las dos fuentes, `process` genera en `data/processed/`:

| Archivo | Contenido |
| --- | --- |
| `pm25_horario.csv` | Serie limpia en formato largo (`estacion`, `fecha`, `pm25`): sin valores negativos ni fechas inválidas; duplicados estación-hora promediados. Pesa unos 50 MB y no se sube a git |
| `pm25_diario.csv` | Promedio diario por estación, con `n_horas` y `valido` (al menos 18 de 24 h) |
| `reporte_unidades.csv` | Mediana y percentil 99 por estación y año; `sospechoso` si la mediana supera 100 µg/m³ |
| `estaciones.csv` | Rango de fechas por estación (y coordenadas, si la fuente las trae; el SIATA no las incluye) |

### 5. Corpus del RAG

```bash
python scripts/download_corpus.py
```

Descarga los 4 documentos a `data/corpus/` y escribe `data/corpus/_fuentes.json` con título, nombre corto y URL de cada uno, que el RAG usa para citar. Si alguna descarga falla, imprime qué bajar a mano.

| Archivo | Documento | Qué aporta al agente |
| --- | --- | --- |
| `res_2254_2017_mads.html` | Resolución 2254 de 2017, MADS | Límite diario de 37 µg/m³, niveles de prevención, alerta y emergencia, e ICA (las tablas son imágenes) |
| `poeca_acuerdo_04_2018_amva.html` | POECA, Acuerdo Metropolitano 04 de 2018 (texto compilado, con las modificaciones del Acuerdo 7 de 2022) | Declaración de niveles (Arts. 16–18) y medidas por nivel (Art. 27) |
| `pigeca_acuerdo_16_2017_amva.html` | PIGECA, Acuerdo Metropolitano 16 de 2017 | Marco de gestión de la calidad del aire 2017–2030 |
| `siata_informe_operacion_2025_01.pdf` | Informe de operación de la Red de Calidad del Aire, enero 2025 (SIATA) | Estaciones, excedencias y la regla del 75 % de datos válidos |

El POECA y el PIGECA vienen del [normograma oficial del AMVA](http://alphasig.metropol.gov.co/normograma/compilacion/docs/a_amva_0004_2018.htm). En los tres documentos normativos, las tablas con umbrales numéricos son imágenes y el RAG no las lee.

### 6. ICA y niveles de episodio

`src/ica.py` ya contiene los puntos de corte para PM2.5 en 24 h, transcritos el 6 nov 2026 de las imágenes de la Res. 2254 de 2017 en la página de la CRA. Conviene cotejarlos con el PDF del Diario Oficial 50.415.

| µg/m³ (24 h) | ICA | Categoría (Tabla 6) | Color (Tabla 5) | Nivel (Tabla 4) |
| --- | --- | --- | --- | --- |
| 0–12 | 0–50 | Buena | Verde | — |
| 13–37 | 51–100 | Aceptable | Amarillo | — |
| 38–55 | 101–150 | Dañina a la salud de grupos sensibles | Naranja | Prevención |
| 56–150 | 151–200 | Dañina a la salud | Rojo | Alerta |
| 151–250 | 201–300 | Muy dañina a la salud | Púrpura | Emergencia |
| 251–500 | 301–500 | Peligrosa | Marrón | Emergencia |

Los límites superiores son inclusivos y los valores en un hueco (p. ej. 12,4) van al rango superior. Así, "Aceptable" termina en 37 µg/m³, el máximo diario permisible. La norma calcula el ICA con medias móviles de 24 h; el proyecto lo aplica al promedio del día calendario.

Distribución en los datos 2019–2026 (60.902 estación-día válidos): Buena 24,4 %, Aceptable 73,0 %, Naranja 2,3 %, Rojo 0,3 %, ningún día Púrpura o Marrón.

### 7. EDA

Abre `notebooks/01_eda.ipynb` y completa la sección de conclusiones.

## Plan C: CSV manual

Si fallan el SIATA y datos.gov.co, descarga un CSV en formato largo (portal de datos abiertos del AMVA u otra fuente oficial), ajusta `COLUMNS` a sus nombres y corre:

```bash
python -m src.data process --raw ruta/al/archivo.csv
```

## Estado de las fuentes

| Fuente | Estado |
| --- | --- |
| Repositorio de Datos SIATA, DOI 10.83041/AUWZWT | **Fuente principal, verificada el 6 nov 2026.** 165 archivos mensuales (2013-01 a 2026-09); descargados y procesados 2019-2026. Sin estación-año con mediana > 100 µg/m³ |
| Coordenadas de estaciones | No vienen en el dataset del SIATA; pendiente para el mapa del dashboard |
| Recurso `g4t8-zkc3` en datos.gov.co | Respaldo. Identificador tomado del [README de BRUMA](https://github.com/saracastrillons/bruma). Sin confirmar |
| Paginación de la API | Según la [documentación oficial de Socrata](https://dev.socrata.com/docs/paging.html): hasta 50 000 filas por página en endpoints 2.0 y `$order=:id` para orden estable |
| Límite diario PM2.5 = 37 µg/m³ | [Res. 2254 de 2017](https://normas.cra.gov.co/gestor/docs/resolucion_minambienteds_2254_2017.htm), Art. 2, parágrafo 1 (verificado) |
| Puntos de corte del ICA y niveles de episodio | Tablas 4, 5 y 6 de la misma resolución, transcritas de sus imágenes en `src/ica.py` (6 nov 2026). Pendiente cotejar con el PDF del Diario Oficial |
| Regla de 18 de 24 horas válidas | Verificada en el informe SIATA de enero 2025 del corpus ("NP: No hay promedio por porcentaje de datos válidos inferior al 75 %") |
| Corpus del RAG | 4 documentos descargados y verificados el 6 nov 2026. El PDF del PIGECA en medellin.gov.co y las páginas del POECA en metropol.gov.co devuelven 404; se usa el normograma del AMVA |

## Estructura actual

```text
calidad-aire/
├── src/
│   ├── config.py      # rutas, columnas, filtros, umbrales
│   ├── data.py        # leer_siata, process (+ inspect/download de respaldo)
│   └── ica.py         # categorías ICA y niveles de episodio (Res. 2254)
├── scripts/
│   ├── download_siata.py   # fuente principal
│   └── download_corpus.py
├── notebooks/01_eda.ipynb
├── tests/test_data.py
├── data/              # raw/ (ignorado en git), processed/, corpus/
└── requirements.txt
```
