# Declaración de uso de IA generativa

Proyecto final SI3003 — Agente de calidad del aire del Valle de Aburrá.

> Este documento lo redactó Claude a partir del registro de trabajo. La sección 3 recoge las respuestas del autor.

## 1. Herramientas usadas

| Herramienta | Para qué |
| --- | --- |
| **Claude** (Anthropic), modelo Claude Opus 5.5, en la aplicación Claude Code (escritorio) | Asistente de programación durante el desarrollo (detalle en la sección 2) |
| **Groq**, modelo `openai/gpt-oss-120b` | Componente del sistema: es el LLM del agente dentro de la app (Parte 2). No se usó para escribir código ni documentos |
| **fastembed** con `paraphrase-multilingual-MiniLM-L12-v2` | Componente del sistema: embeddings del RAG. No es IA generativa |

## 2. Uso de Claude en el desarrollo

Claude trabajó con acceso a la carpeta del proyecto y a una terminal, a pedido del autor y sesión por sesión según el plan de acción.

- **Plan de acción:** el plan del proyecto (sesiones, alcance, riesgos y rúbrica) se elaboró con ayuda de Claude.

- **Datos:** localizó y verificó las fuentes. Identificó el Repositorio de Datos SIATA, los metadatos de estaciones, el normograma del AMVA para el POECA y el PIGECA, y los términos de uso del repositorio. Escribió los scripts de descarga y limpieza, y el EDA.
- **Normativa:** ayudó a localizar la Res. 2254 en la compilación de la Alcaldía de Bogotá (SISJUR, https://www.alcaldiabogota.gov.co/sisjur/normas/Norma1.jsp?i=82634).
- **Parte 1:** propuso el diseño (variables, partición temporal, líneas base y bootstrap) y escribió la regresión logística multinomial en NumPy, las métricas y la prueba de cordura contra scikit-learn. También señaló que la selección de hiperparámetros se cambió después de mirar la prueba una vez; está documentado en el README.
- **Parte 2:** escribió el cliente LLM, las herramientas, el agente con su loop y T_MAX, el RAG (fragmentación, índice FAISS y búsqueda híbrida con BM25) y la evaluación del agente.
- **Dashboard:** escribió la app de Streamlit.
- **Depuración:** encontró y corrigió errores probando el sistema, por ejemplo:
  - fragmentos que mezclaban niveles del Art. 27;
  - el mapa con todas las estaciones en el mismo punto;
  - las citas con corchetes 【】;
  - una respuesta que mezclaba efectos de dos categorías del ICA (detectada en revisión manual);
  - el riesgo de que una versión nueva de Streamlit rompiera la app (versiones fijadas en `requirements.txt`).
- **Ajustes a partir de las pruebas del autor:** implementó la regla de alcance del agente (dentro del tema, fuera de tema, ambigua), agregó las preguntas de evaluación P13–P15 y repitió las corridas de evaluación (v4 a v6).
- **Documentación:** redactó el README (incluida la arquitectura con diagramas Mermaid), esta declaración, el informe, `eval/RESULTADOS.md`.

## 3. Qué hizo y verificó el autor

- Definió el problema y el alcance. El plan de acción se elaboró con ayuda de Claude y el autor lo usó como guía.
- Tomó las decisiones en cada sesión: proveedor del LLM, mejorar la selección del modelo, cuándo subir a GitHub, entre otras.
- Revisó los resultados (métricas, respuestas del agente e informe). Auditó el código generado por Claude.
- **Sometió el agente a pruebas propias para optimizarlo.** Usando la app, le hizo preguntas fuera de tema ("¿quién es Perla?", "¿quién es Judas?", "¿cuánto es 4x4?", "¿cuál es la capital de Bangladesh?"), una pregunta sobre el aire con premisa dudosa ("¿los gatos bengalíes afectan el aire de Medellín?") y una sobre un dato que el sistema no tiene ("¿cómo está el clima en Aranjuez?"). Así encontró que el agente respondía temas ajenos y que, tras el primer ajuste, rechazaba preguntas válidas sobre el aire. A partir de esos hallazgos se redefinió la regla de alcance y se agregaron las preguntas P13, P14 y P15 a la evaluación. También pidió a conocidos preguntas extra para probarlo ("hazme una muestra gráfica de en qué partes del área metropolitana lloverá en las próximas 3 horas" y "¿en promedio cuánto demora en subir la sensación térmica después de un aguacero?"): en ambas el agente respondió que solo puede ayudar con calidad del aire y propuso preguntas que sí puede responder, sin inventar lluvia ni temperaturas, así que no hizo falta otro ajuste.
- Detectó que el informe era corto y que las preguntas de prueba no eran fáciles de revisar; a partir de eso se agregaron al informe la tabla por pregunta y una captura del dashboard, y se creó `eval/RESULTADOS.md`.
- Pidió agregar la arquitectura del proyecto al README.
- Revisó las entradas de las Tablas 4, 5 y 6 de la Res. 2254 de la Alcaldía de Bogotá (SISJUR, https://www.alcaldiabogota.gov.co/sisjur/normas/Norma1.jsp?i=82634), fuente independiente de la página de la CRA usada para verificar: coinciden en las tres tablas. No se pudo descargar el PDF del Diario Oficial 50.415.
- Revisó el informe.
- Grabó el video demostrativo: https://www.youtube.com/watch?v=janrE4cZDAQ

## 4. Cómo se verificó lo producido con IA

- 68 pruebas automáticas (`pytest`), entre ellas:
  - la comparación del gradiente con diferencias finitas;
  - una prueba de no fuga de información en las variables;
  - pruebas del agente con un LLM simulado.
- Prueba de cordura del clasificador contra scikit-learn: 100 % de las predicciones iguales.
- Revisión manual de que las citas del agente estén en el texto citado.
- Cotejo de las tablas transcritas de la Res. 2254 contra una fuente independiente (SISJUR).
- Evaluación del agente con 15 preguntas, verificación automática contra los datos y revisión manual de respuestas (`eval/`).
- Prueba de reproducibilidad en un clon limpio del repositorio.

## 5. Integridad académica

- El código es propio del proyecto. No se copió código del repositorio BRUMA ni de otros proyectos. BRUMA solo sirvió, al inicio, para ubicar una posible fuente de datos, que al final no se usó.
- Los datos son del SIATA/AMVA y se citan según sus términos de uso (borrador, abril 2026): uso académico, sin fines comerciales, con atribución.
- Los resultados se reportan tal como salieron, incluidos los que no favorecen al modelo: el clasificador empata con la persistencia en F1 macro.
