# Declaración de uso de IA generativa

Proyecto final SI3003 — Agente de calidad del aire del Valle de Aburrá.

> Este documento lo redactó un asistente de IA a partir del registro de trabajo, y lo revisa y completa el autor. Los puntos marcados **[COMPLETA]** solo los puede responder el autor.

## 1. Herramientas usadas

| Herramienta | Para qué |
| --- | --- |
| **Claude** (Anthropic), modelo Claude Opus 5.5, en la aplicación Claude Code (escritorio) | Asistente de programación durante el desarrollo (detalle en la sección 2) |
| **Groq**, modelo `openai/gpt-oss-120b` | Componente del sistema: es el LLM del agente dentro de la app (Parte 2). No se usó para escribir código ni documentos |
| **fastembed** con `paraphrase-multilingual-MiniLM-L12-v2` | Componente del sistema: embeddings del RAG. No es IA generativa |
| [COMPLETA: otras herramientas, si las hubo; p. ej. para el plan de acción o el video] | |

## 2. Uso de Claude en el desarrollo

Claude trabajó con acceso a la carpeta del proyecto y a una terminal, a pedido del autor y sesión por sesión según el plan de acción.

- **Datos:** localizó y verificó las fuentes. Identificó el Repositorio de Datos SIATA, los metadatos de estaciones, el normograma del AMVA para el POECA y el PIGECA, y los términos de uso del repositorio. Escribió los scripts de descarga y limpieza, y el EDA.
- **Normativa:** transcribió las Tablas 4, 5 y 6 de la Res. 2254 a partir de sus imágenes. El autor debe cotejarlas con el PDF oficial (ver sección 3).
- **Parte 1:** propuso el diseño (variables, partición temporal, líneas base y bootstrap) y escribió la regresión logística multinomial en NumPy, las métricas y la prueba de cordura contra scikit-learn. También señaló que la selección de hiperparámetros se cambió después de mirar la prueba una vez; está documentado en el README.
- **Parte 2:** escribió el cliente LLM, las herramientas, el agente con su loop y T_MAX, el RAG (fragmentación, índice FAISS y búsqueda híbrida con BM25) y la evaluación del agente.
- **Dashboard:** escribió la app de Streamlit y la probó en un navegador.
- **Depuración:** encontró y corrigió errores probando el sistema, por ejemplo:
  - fragmentos que mezclaban niveles del Art. 27;
  - el mapa con todas las estaciones en el mismo punto;
  - las citas con corchetes 【】;
  - el límite de tokens de Groq.
- **Documentación:** redactó el README, esta declaración, el borrador del informe y el guion del video.

## 3. Qué hizo y verificó el autor

- Definió el problema, el alcance y el plan de acción. [COMPLETA: indica si el plan se hizo con ayuda de IA]
- Tomó las decisiones en cada sesión: proveedor del LLM, mejorar la selección del modelo, cuándo subir a GitHub, entre otras. [COMPLETA]
- Revisó el código y los resultados. [COMPLETA: qué revisaste y cómo; qué cambiaste a mano]
- Cotejó las tablas transcritas de la Res. 2254 con el PDF oficial del Diario Oficial 50.415. [COMPLETA: sí/no y resultado]
- Revisó y ajustó el informe. [COMPLETA]
- Grabó el video. [COMPLETA]

## 4. Cómo se verificó lo producido con IA

- 65 pruebas automáticas (`pytest`), entre ellas:
  - la comparación del gradiente con diferencias finitas;
  - una prueba de no fuga de información en las variables;
  - pruebas del agente con un LLM simulado.
- Prueba de cordura del clasificador contra scikit-learn: 100 % de las predicciones iguales.
- Revisión manual de que las citas del agente estén en el texto citado.
- Evaluación del agente con 12 preguntas y verificación automática contra los datos (`eval/`).
- Prueba de reproducibilidad en un clon limpio del repositorio.

## 5. Integridad académica

- El código es propio del proyecto. No se copió código del repositorio BRUMA ni de otros proyectos. BRUMA solo sirvió, al inicio, para ubicar una posible fuente de datos, que al final no se usó.
- Los datos son del SIATA/AMVA y se citan según sus términos de uso (borrador, abril 2026): uso académico, sin fines comerciales, con atribución.
- Los resultados se reportan tal como salieron, incluidos los que no favorecen al modelo: el clasificador empata con la persistencia en F1 macro.
