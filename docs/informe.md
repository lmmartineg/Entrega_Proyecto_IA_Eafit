---
titulo: "Pronóstico de PM2.5 y agente de consulta normativa para el Valle de Aburrá"
subtitulo: "Proyecto final SI3003 Inteligencia Artificial · EAFIT · noviembre de 2026"
autor: "[COMPLETA: nombre del autor]"
---

> Herramienta analítica con fines académicos; **no es consejo médico**. Código y datos procesados: https://github.com/lmmartineg/Entrega_Proyecto_IA_Eafit

## 1. Problema

El Valle de Aburrá sufre episodios de contaminación por material particulado fino (PM2.5), sobre todo entre febrero y abril y entre octubre y noviembre. La Res. 2254 de 2017 fija un máximo diario de 37 µg/m³ y define el Índice de Calidad del Aire (ICA) y los niveles de prevención, alerta y emergencia. El POECA (Acuerdo Metropolitano 04 de 2018) define qué medidas se activan en cada nivel.

El proyecto responde dos preguntas: **¿qué categoría de PM2.5 tendrá mañana una estación?** y **¿qué implica esa categoría según la norma?** Para eso combina dos partes:

- **Parte 1:** un clasificador implementado desde cero.
- **Parte 2:** un agente LLM que usa ese clasificador, los datos medidos y un RAG sobre la normativa, citando sus fuentes.

La combinación ocurre en las herramientas del agente: en una misma conversación llama al modelo propio y a la normativa.

## 2. Datos

- **Mediciones:** PM2.5 horario del Repositorio de Datos SIATA (doi:10.83041/AUWZWT), de enero de 2019 a septiembre de 2026, para 30 estaciones.
- **Estaciones:** nombre, coordenadas y municipio del dataset "Información de la Red de Calidad del Aire" (doi:10.83041/XTI3FH).
- **Promedio diario:** se calcula por estación y día. El día se acepta solo si tiene al menos 18 de 24 horas con dato, la regla del 75 % que aplican los informes del SIATA. Resultan 62.593 estación-día, de los cuales el 97,3 % son válidos.
- **Unidades:** se revisaron porque otra fuente pública (datos.gov.co) tiene valores fuera de escala entre 2020 y 2023. En estos datos no aparece ese problema: las medianas anuales van de 6,5 a 30 µg/m³.
- **Etiqueta:** la categoría ICA del día siguiente (Tabla 6 de la Res. 2254), agrupada en tres clases. "Naranja o peor" (> 37 µg/m³) es solo el 2,6 % de los días.

| Clase | Rango (µg/m³) | % de días 2019–2026 |
| --- | --- | --- |
| 0 · Buena | 0–12 | 24,4 % |
| 1 · Aceptable | 13–37 | 73,0 % |
| 2 · Dañina para grupos sensibles o peor | > 37 | 2,6 % |

## 3. Parte 1: clasificador

**Variables (sin fuga de información).** Al cierre del día *t* se predice la clase de *t+1* con:

- el PM2.5 de los días *t* a *t*−6 y la media de 14 días, en escala log(1+x);
- el promedio del valle en *t*;
- el día de la semana y el mes de *t+1*.

Una prueba automática comprueba que cambiar el dato de *t+1* no altera ninguna variable de la fila *t*. Quedan 58.249 filas con etiqueta.

**Modelo.** Regresión logística multinomial (softmax) escrita en NumPy:

- entropía cruzada con penalización L2 y pesos de clase;
- descenso de gradiente con aceleración de Nesterov y paso 1/L, donde L es la cota de Böhning de la curvatura;
- el gradiente se verificó contra diferencias finitas;
- scikit-learn, con la misma penalización y los mismos datos, da el 100 % de predicciones iguales; solo se usó como prueba de cordura.

**Validación temporal.**

- **Desarrollo:** 2019 a 2023.
- **Prueba:** enero de 2024 a septiembre de 2026, con 20.777 estación-día.
- **Selección de L2 y pesos:** validación con ventana creciente. Cada año de 2021 a 2023 se predice con un modelo entrenado en los años anteriores.
- **Elegido:** L2 = 0,01 con pesos "raíz", la raíz cuadrada de los pesos balanceados.

La prueba no se tomó del último año completo, como decía el plan, porque 2025 tiene solo 7 casos de la clase 2.

**Resultados en prueba:**

| Método | Exactitud | F1 macro | F1 clase 2 | Precisión / sensibilidad clase 2 |
| --- | --- | --- | --- | --- |
| Regresión softmax (propia) | **0,808** | 0,665 | 0,428 | 0,33 / **0,60** |
| Persistencia (mañana = hoy) | 0,796 | **0,685** | **0,540** | **0,54** / 0,54 |
| Clase mayoritaria | 0,703 | 0,275 | 0 | 0 / 0 |

Diferencias del modelo frente a la persistencia, con bootstrap por bloques de 7 días (2.000 repeticiones, IC 95 %):

- **Exactitud:** el modelo es mejor, +0,012 [+0,001; +0,021], sobre todo porque reconoce mejor los días "Buena".
- **F1 macro:** empate, −0,021 [−0,055; +0,010].
- **F1 de la clase 2:** la persistencia es mejor, −0,112 [−0,211; −0,024]. El modelo detecta más episodios, pero con más falsas alarmas: solo 1 de cada 3 de sus alertas es correcta.

Que un modelo lineal sobre rezagos no supere a la persistencia es coherente con la fuerte autocorrelación del PM2.5 diario (0,82 a un día).

**Nota de integridad.** La primera versión elegía hiperparámetros con un solo año de validación (2023, con 57 casos de la clase 2) y obtuvo F1 macro 0,635. Se cambió a la validación con ventana creciente después de ver la prueba una vez. El cambio usa solo datos de desarrollo.

## 4. Parte 2: agente con herramientas y RAG

**Agente.** Usa un loop manual con T_MAX = 5 iteraciones. En cada iteración el LLM responde o pide herramientas. Sus resultados, o sus errores (estación inexistente, fecha sin datos, JSON inválido), vuelven como observación. Si se agotan las iteraciones, una última llamada sin herramientas cierra la respuesta. El LLM es `openai/gpt-oss-120b` en Groq, a través de una API compatible con OpenAI, y se puede definir un proveedor alterno.

Herramientas:

1. **`get_pm25(estación, fecha)`:** promedio diario, categoría ICA, nivel de episodio y comparación con 37 µg/m³.
2. **`predict_tomorrow(estación, fecha)`:** la clase y las probabilidades del modelo propio, junto con su desempeño real en prueba.
3. **`search_docs(consulta)`:** fragmentos de la normativa con su cita.

**RAG.** El corpus tiene cuatro documentos oficiales: la Res. 2254, el POECA compilado con sus modificaciones de 2022, el PIGECA y un informe de operación del SIATA. Se suma la transcripción de las Tablas 4 a 6 de la Res. 2254, que en la fuente son imágenes.

- **Fragmentación:** 599 fragmentos de unas 110 palabras, divididos por artículo (o por página en el PDF), para que la cita sea [documento, artículo].
- **Corrección importante:** la primera versión mezclaba niveles. Atribuía al nivel de alerta medidas que el POECA define para el de emergencia, porque el Art. 27 tiene 3.600 palabras. Ahora se divide por nivel, por ejemplo "Art. 27, nivel de alerta".
- **Índice:** embeddings multilingües locales (MiniLM, 384 dimensiones) en FAISS, combinados con BM25 mediante fusión por rango recíproco.
- **Recuperación:** la cita esperada aparece entre las 8 primeras en 12 de 12 consultas de ajuste, frente a 10 de 12 solo con embeddings. Es una cifra optimista, porque esas consultas sirvieron para ajustar la búsqueda.
- **Respuestas:** la instrucción de sistema obliga a citar y a decir "No encontré esto en los documentos consultados" cuando no haya respuesta.

**Dashboard (Streamlit).** Incluye:

- selector de estación y fecha;
- mapa de estaciones coloreado por ICA;
- serie de tiempo con el límite de 37 µg/m³;
- tarjeta de predicción, comparada con lo que pasó realmente cuando ya hay dato;
- chat con el agente, que muestra cada herramienta usada y sus errores.

## 5. Evaluación del agente

Se escribieron 12 preguntas fijas, distintas de las de ajuste del RAG:

- 3 de datos;
- 3 de pronóstico;
- 4 de normativa;
- 1 combinada;
- 1 fuera del corpus.

La evaluación mide automáticamente:

- si se llamó la herramienta esperada;
- si la respuesta trae cita;
- si las citas son válidas, es decir, si cada cita corresponde a un fragmento que de verdad devolvió la búsqueda;
- si el contenido es correcto: el PM2.5 real calculado al correr, la clase predicha por el modelo y frases esperadas;
- si incluye el aviso médico.

| Corrida | Herramienta | Con cita | Citas válidas | Contenido | Errores de API |
| --- | --- | --- | --- | --- | --- |
| v1 | 75 % | 0 % | — | 25 % | 9/12 (límite de tokens) |
| v2 | 100 % | 60 % | 100 % | 100 % | 0 |
| v3 | 100 % | 100 % | 100 % | 100 % | 0 |

Cada corrida llevó a una corrección:

- **v1:** el error fue de infraestructura. Ahora el cliente espera el tiempo que indica la API y reintenta.
- **v2:** el modelo citaba con corchetes lenticulares (caracteres U+3010 y U+3011) en lugar de [ ]. Ahora la instrucción pide corchetes normales y una limpieza los normaliza siempre igual.

## 6. Limitaciones

- No hay datos en tiempo real: "hoy" es el 30 sep 2026, y la app lo dice.
- El ICA se aplica al promedio del día calendario, mientras que la norma usa medias móviles de 24 h.
- La clase 2 es rara: el modelo no supera a la persistencia en F1 macro y tiene muchas falsas alarmas.
- Las tablas de la Res. 2254 se transcribieron de imágenes. [COMPLETA: resultado del cotejo con el PDF oficial]
- El 100 % de la evaluación del agente es una sola corrida sobre 12 preguntas escritas por el equipo; el LLM puede variar entre corridas.

**Trabajo futuro:** variables meteorológicas y un modelo no lineal o un HMM, evaluar con más preguntas y varias corridas, e incorporar datos del SIATA en tiempo real.

**Uso de IA generativa:** ver `DECLARACION_IA.md`.

**Referencias:**

- SIATA (2026). Material Particulado PM2.5, doi:10.83041/AUWZWT.
- SIATA. Información de la Red de Calidad del Aire, doi:10.83041/XTI3FH.
- MADS, Resolución 2254 de 2017.
- AMVA, Acuerdos Metropolitanos 04 de 2018 (POECA) y 16 de 2017 (PIGECA).
