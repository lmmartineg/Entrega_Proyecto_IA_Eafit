# Evaluación del agente: resultados por pregunta

Corrida **v6** del 2026-11-12 · LLM: groq (`openai/gpt-oss-120b`) · 15 preguntas fijas (`eval/preguntas.yaml`). Se genera con `python eval/run_eval.py`.

| Métrica | Resultado |
| --- | --- |
| Herramienta correcta | 93 % |
| Cita presente (en las que la exigen) | 100 % |
| Citas válidas (no inventadas) | 100 % |
| Contenido verificado | 93 % |
| Aviso médico | 100 % |

**Cómo leer la tabla.** *Herramienta*: se llamaron las herramientas esperadas. *Cita*: la respuesta cita [documento, ubicación] (solo se exige en normativa). *Válidas*: cada cita corresponde a un fragmento que devolvió `search_docs`. *Contenido*: el valor de PM2.5 coincide con el dato real, se nombra la clase del modelo o aparecen las frases esperadas.

| ID | Categoría | Pregunta | Esperada | Llamadas | Herramienta | Cita | Válidas | Contenido |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P01 | datos | ¿Cuál fue el PM2.5 promedio en la estación MED-ARAN el 15 de marzo de 2020? | `get_pm25` | `get_pm25` | ✅ | no se exige | ✅ | ✅ |
| P02 | datos | ¿Cómo está el aire hoy en Itagüí, en la estación Casa de Justicia? | `get_pm25` | `get_pm25` | ✅ | no se exige | — | ✅ |
| P03 | datos | ¿Se superó el límite diario de PM2.5 en CEN-TRAF el 6 de febrero de 2019? | `get_pm25` | `get_pm25` | ✅ | no se exige | — | ✅ |
| P04 | pronostico | ¿Qué calidad del aire se espera mañana en Santa Elena? | `predict_tomorrow` | `predict_tomorrow` | ✅ | no se exige | — | ✅ |
| P05 | pronostico | Según el modelo, ¿qué clase de PM2.5 se esperaba en MED-ARAN para el día siguiente al 14 de marzo de 2020? | `predict_tomorrow` | `predict_tomorrow` | ✅ | no se exige | — | ✅ |
| P06 | pronostico | ¿Habrá mañana un día por encima de la norma en Bello, en la I.E. Fernando Vélez? | `predict_tomorrow` | `predict_tomorrow` | ✅ | no se exige | — | ✅ |
| P07 | normativa | ¿Qué medidas se aplican cuando se declara el nivel de emergencia en el Valle de Aburrá? | `search_docs` | `search_docs` | ✅ | ✅ | ✅ | ✅ |
| P08 | normativa | ¿A partir de qué concentración de PM2.5 en 24 horas se declara el nivel de prevención? | `search_docs` | `search_docs` | ✅ | ✅ | ✅ | ✅ |
| P09 | normativa | ¿Qué significa para la salud que el índice de calidad del aire esté en rojo? | `search_docs` | `search_docs` | ✅ | ✅ | ✅ | ✅ |
| P10 | normativa | Una vez declarado un nivel de prevención, alerta o emergencia, ¿durante cuánto tiempo mínimo se mantienen las medidas? | `search_docs` | `search_docs` | ✅ | ✅ | ✅ | ✅ |
| P11 | combinada | El 15 de marzo de 2020, ¿la concentración en CEN-TRAF correspondía a algún nivel de episodio? ¿Qué dice la norma sobre ese nivel? | `get_pm25,search_docs` | `get_pm25,search_docs` | ✅ | ✅ | ✅ | ✅ |
| P12 | fuera_del_corpus | ¿Qué dice la normativa sobre los límites de ruido para motocicletas? | `search_docs` | `ninguna` | ❌ | no se exige | — | ❌ |
| P13 | fuera_de_tema | ¿Cuál es la capital de Bangladesh? | `—` | `ninguna` | ✅ | no se exige | — | ✅ |
| P14 | premisa_dudosa | ¿Los gatos bengalíes afectan el aire de Medellín? | `search_docs` | `search_docs` | ✅ | no se exige | ✅ | ✅ |
| P15 | fuera_de_datos | ¿Cómo está el clima en Aranjuez? | `—` | `ninguna` | ✅ | no se exige | — | ✅ |

## Respuestas del agente

<details><summary><b>P01</b> · ¿Cuál fue el PM2.5 promedio en la estación MED-ARAN el 15 de marzo de 2020?</summary>

*Herramientas:* `get_pm25` · *comprobación:* PM2.5 real 50.2 encontrado

> El 15 de marzo de 2020 la estación MED‑ARAN (I.E. Ciro Mendia, Aranjuez) registró un promedio diario de PM2.5 de **50.2 µg/m³**, con categoría ICA “Dañina a la salud de grupos sensibles” y superó el límite diario de 37 µg/m³ establecido por la normativa [Res. 2254 de 2017, Art. 2].  
> 
> Herramienta analítica, no es consejo médico.

</details>

<details><summary><b>P02</b> · ¿Cómo está el aire hoy en Itagüí, en la estación Casa de Justicia?</summary>

*Herramientas:* `get_pm25` · *comprobación:* PM2.5 real 17.3 encontrado; /tiempo real|más reciente|último dato|últim[oa] (dato|registro|medición)/ sí

> Hoy (último dato disponible 30‑09‑2026) la estación Casa de Justicia de Itagüí registró un PM2.5 promedio de **17.3 µg/m³**, categoría ICA **Aceptable** (color amarillo) y no supera el límite diario de 37 µg/m³. [SIATA, 2026]  
> 
> Herramienta analítica, no es consejo médico.

</details>

<details><summary><b>P03</b> · ¿Se superó el límite diario de PM2.5 en CEN-TRAF el 6 de febrero de 2019?</summary>

*Herramientas:* `get_pm25` · *comprobación:* PM2.5 real 41.1 encontrado; /\bs[ií]\b|super[oó]|supera/ sí

> Sí, el 6 de febrero de 2019 la estación CEN‑TRAF (Centro, Medellín) registró un promedio diario de PM2.5 de 41.1 µg/m³, superando el límite diario de 37 µg/m³ establecido por la normativa. Herramienta analítica, no es consejo médico.

</details>

<details><summary><b>P04</b> · ¿Qué calidad del aire se espera mañana en Santa Elena?</summary>

*Herramientas:* `predict_tomorrow` · *comprobación:* clase del modelo 'Buena' nombrada

> Se prevé que la calidad del aire en Santa Elena (estación MED‑SELE) sea **Buena** para el día siguiente al último dato disponible (30 de septiembre de 2026). Las probabilidades son 91,5 % para “Buena”, 8,5 % para “Aceptable” y 0 % para “Dañina para grupos sensibles o peor”.  
> 
> Esta predicción corresponde a la clase de PM2.5 que el modelo propio del proyecto estima para el día siguiente al último registro (30‑09‑2026); no representa un pronóstico en tiempo real para “mañana”.  
> 
> Herramienta analítica, no es consejo médico.

</details>

<details><summary><b>P05</b> · Según el modelo, ¿qué clase de PM2.5 se esperaba en MED-ARAN para el día siguiente al 14 de marzo de 2020?</summary>

*Herramientas:* `predict_tomorrow` · *comprobación:* clase del modelo 'Dañina para grupos sensibles o peor' nombrada

> Según el modelo, la clase de PM2.5 esperada para MED‑ARAN el día siguiente al 14 de marzo de 2020 era **“Dañina para grupos sensibles o peor”**.  
> Herramienta analítica, no es consejo médico.

</details>

<details><summary><b>P06</b> · ¿Habrá mañana un día por encima de la norma en Bello, en la I.E. Fernando Vélez?</summary>

*Herramientas:* `predict_tomorrow` · *comprobación:* clase del modelo 'Buena' nombrada

> Según la predicción del modelo para la estación **BEL-FEVE (I.E. Fernando Vélez, Bello)**, el día siguiente se clasifica como **“Buena”** (≤ 12 µg/m³), por lo que no se espera que supere la norma de 37 µg/m³. Las probabilidades son: Buena 51 %, Aceptable 49 %, Dañina 0 %.  
> 
> Herramienta analítica, no es consejo médico.

</details>

<details><summary><b>P07</b> · ¿Qué medidas se aplican cuando se declara el nivel de emergencia en el Valle de Aburrá?</summary>

*Herramientas:* `search_docs` · *comprobación:* /emergencia/ sí

> Cuando se declara **nivel de emergencia (Nivel IV)** en el Área Metropolitana del Valle de Aburrá se aplican, según el Acuerdo POECA 04 de 2018, las siguientes medidas:
> 
> - **Comunicación pública**: se informa a la población a través de los medios de comunicación sobre la ocurrencia del episodio, el nivel declarado y las medidas que se aplicarán; se difunden los comunicados oficiales y la evolución del ICA en cada municipio [POECA (Acuerdo Metropolitano 04 de 2018), Art. 27, nivel de emergencia].  
> - **Coordinación institucional**: se convoca a los Consejos Metropolitanos de Gestión del Riesgo de Desastres, Ambiental y de Movilidad para coordinar la implementación de las medidas [POECA (Acuerdo Metropolitano 04 de 2018), Art. 27, nivel de emergencia].  
> - **Reducción de emisiones industriales**:  
>   * Las empresas que no hayan demostrado cumplimiento de los estándares nacionales para material particulado no pueden operar [POECA (Acuerdo Metropolitano 04 de 2018), Art. 27, nivel de emergencia].  
>   * Se envía comunicación oficial a cada empresa indicando las fuentes que serán objeto de restricción [POECA (Acuerdo Metropolitano 04 de 2018), Art. 27, nivel de emergencia].  
>   * Las empresas no deben realizar mantenimiento preventivo de los equipos de control de emisiones durante los periodos regulares de aplicación del protocolo (febrero‑abril y octubre‑noviembre) [POECA (Acuerdo Metropolitano 04 de 2018), Art. 27, nivel de emergencia].  
> - **Restricción de transporte y movilidad**: se implementa la restricción total a la circulación de automóviles particulares y de motos de 2 y 4 tiempos en el horario extendido [POECA (Acuerdo Metropolitano 04 de 2018), Art. 27, nivel de emergencia].
> 
> Estas acciones se aplican por un mínimo de tres (3) días consecutivos y se mantienen hasta que los informes técnicos del SIATA y del GECA indiquen que ya no es necesario [POECA (Acuerdo Metropolitano 04 de 2018), Art. 17].
> 
> **Herramienta analítica, no es consejo médico.**

</details>

<details><summary><b>P08</b> · ¿A partir de qué concentración de PM2.5 en 24 horas se declara el nivel de prevención?</summary>

*Herramientas:* `search_docs` · *comprobación:* /\b38\b/ sí

> El nivel de prevención se declara cuando la concentración promedio de PM2.5 en 24 horas está entre **38 µg/m³ y 55 µg/m³** [Res. 2254 de 2017, Tabla 4] [POECA (Acuerdo Metropolitano 04 de 2018), Art. 2]  
> 
> Herramienta analítica, no es consejo médico.

</details>

<details><summary><b>P09</b> · ¿Qué significa para la salud que el índice de calidad del aire esté en rojo?</summary>

*Herramientas:* `search_docs` · *comprobación:* /todos|toda la población/ sí

> El ICA en color **rojo** corresponde al rango **151‑200** y se clasifica como **“Dañina para la salud”**. En este nivel **todos los individuos pueden comenzar a experimentar efectos sobre la salud**, y los grupos sensibles (personas con enfermedades respiratorias o cardiacas, niños, adultos mayores, etc.) pueden presentar **efectos más graves**[Res. 2254 de 2017, Tabla 5].  
> 
> Para el contaminante PM2.5, el rango que genera este color es **de 56 a 150 µg/m³**[Res. 2254 de 2017, Tabla 6].  
> 
> Herramienta analítica, no es consejo médico.

</details>

<details><summary><b>P10</b> · Una vez declarado un nivel de prevención, alerta o emergencia, ¿durante cuánto tiempo mínimo se mantienen las medidas?</summary>

*Herramientas:* `search_docs` · *comprobación:* /tres \(3\) días|tres días|3 días/ sí

> Una vez declarados los niveles de prevención, alerta o emergencia, las medidas deben mantenerse al menos tres (3) días consecutivos, y pueden prolongarse hasta que los informes técnicos del SIATA y del GECA lo consideren necesario[POECA (Acuerdo Metropolitano 04 de 2018), Art. 17].  
> Herramienta analítica, no es consejo médico.

</details>

<details><summary><b>P11</b> · El 15 de marzo de 2020, ¿la concentración en CEN-TRAF correspondía a algún nivel de episodio? ¿Qué dice la norma sobre ese nivel?</summary>

*Herramientas:* `get_pm25,search_docs` · *comprobación:* PM2.5 real 52.8 encontrado; /prevenci[oó]n/ sí

> El 15 de marzo de 2020 la estación CEN‑TRAF registró un promedio diario de PM2.5 de 52.8 µg/m³, lo que la ubica dentro del **nivel de episodio “Prevención”** según la Res. 2254 de 2017 [Res. 2254 de 2017, Art. 10].
> 
> La norma establece que el nivel de **Prevención** corresponde a concentraciones promedio de 24 h entre **38 y 55 µg/m³** para PM2.5[Res. 2254 de 2017, Tabla 4], y que dichas declaratorias pueden basarse en los datos de un punto de monitoreo cuando este presenta condiciones para el nivel correspondiente[Res. 2254 de 2017, Art. 12].
> 
> Herramienta analítica, no es consejo médico.

</details>

<details><summary><b>P12</b> · ¿Qué dice la normativa sobre los límites de ruido para motocicletas?</summary>

*Herramientas:* `ninguna` · *comprobación:* /no encontr|no (se )?(menciona|incluye|contiene|aborda|aparece)|no hay información/ NO

> Lo siento, solo puedo ayudar con preguntas sobre calidad del aire y temas ambientales del Valle de Aburrá. Por ejemplo, puedo responder a consultas como:
> 
> - ¿Cuál es la concentración promedio de PM2.5 en la estación MED‑ARAN?
> - ¿Qué niveles de alerta del ICA están vigentes en Medellín?
> - ¿Qué medidas recomienda la normativa para reducir las emisiones de partículas en la zona?
> 
> Si tienes alguna duda sobre calidad del aire, estaré encantado de ayudar.  
> Herramienta analítica, no es consejo médico.

</details>

<details><summary><b>P13</b> · ¿Cuál es la capital de Bangladesh?</summary>

*Herramientas:* `ninguna` · *comprobación:* /calidad del aire/ sí; sin /daca|dhaka/ sí

> Lo siento, solo puedo ayudar con preguntas sobre la calidad del aire y la salud ambiental en el Valle de Aburrá (Medellín y sus municipios). Por ejemplo, puedo responder a consultas como:
> 
> - ¿Cuál es el nivel actual de PM2.5 en la estación MED‑ALTA?
> - ¿Qué medidas se recomiendan cuando el ICA está en categoría “roja”?
> - ¿Cuál es el pronóstico de la calidad del aire para mañana en Bello?
> 
> Si tienes alguna duda sobre esos temas, estaré encantado de ayudar.  
> Herramienta analítica, no es consejo médico.

</details>

<details><summary><b>P14</b> · ¿Los gatos bengalíes afectan el aire de Medellín?</summary>

*Herramientas:* `search_docs` · *comprobación:* /no encontr|no hay (evidencia|información|relación)|no (se )?(menciona|respalda|documenta)/ sí; sin /solo puedo ayudar/ sí

> No encontré en los documentos evidencia de que los gatos bengalíes afecten el aire de Medellín. Los fragmentos consultados describen las fuentes de emisión en el Valle de Aburrá (principalmente tráfico rodado y fuentes fijas) y la red de monitoreo de calidad del aire, pero no mencionan a los gatos ni a mascotas como emisores o influenciadores de la calidad del aire [PIGECA (Acuerdo Metropolitano 16 de 2017), Secc. 3].  
> 
> Herramienta analítica, no es consejo médico.

</details>

<details><summary><b>P15</b> · ¿Cómo está el clima en Aranjuez?</summary>

*Herramientas:* `ninguna` · *comprobación:* /calidad del aire/ sí; sin /\d+\s*°|grados|soleado|nublado|llov|lluvi|temperatura de/ sí

> Lo siento, solo puedo ayudar con preguntas sobre la calidad del aire en el Valle de Aburrá. Por ejemplo, puedo responder:
> 
> - ¿Cuál es la concentración de PM2.5 en la estación de Aranjuez?
> - ¿Qué nivel de alerta del ICA está vigente en Medellín hoy?
> - ¿Qué medidas se recomiendan cuando el ICA está en categoría “Dañina”?
> 
> Si tienes alguna duda sobre la calidad del aire, estaré encantado de ayudar.  
> Herramienta analítica, no es consejo médico.

</details>
