# Guion del video de respaldo (3–4 minutos)

Plan B de la demo, por si la API del LLM o la conexión fallan el día de la presentación.

**Antes de grabar:**

1. Cierra pestañas y notificaciones.
2. Corre `python -m src.llm` para confirmar que Groq responde.
3. Abre `streamlit run app.py` en pantalla completa, con el tema que prefieras.
4. Ten listas las preguntas de abajo, copiadas en un bloc de notas.
5. Espera unos 20 s entre preguntas del chat: con el plan gratuito de Groq, varias seguidas pueden tardar más por el límite de tokens.

| Tiempo | En pantalla | Qué decir (idea, no texto literal) |
| --- | --- | --- |
| 0:00–0:20 | Título del dashboard y aviso | El problema: episodios de PM2.5 en el Valle de Aburrá, la norma de 37 µg/m³ y dos preguntas, qué categoría habrá mañana y qué implica según la norma. Es una herramienta analítica, no consejo médico |
| 0:20–0:50 | Cambia la fecha a **2020-03-15** con el calendario; muestra el mapa | Datos del SIATA de 30 estaciones (2019–2026). Ese día casi todo el valle estuvo en naranja. Pasa el cursor por una estación y muestra el tooltip |
| 0:50–1:20 | Clic en **CEN-TRAF** en el mapa; serie de tiempo | La serie diaria con el límite de 37 µg/m³. La selección funciona desde el mapa |
| 1:20–1:55 | Tarjetas: predicción para 2020-03-16 y "lo que pasó realmente" | Parte 1: regresión logística multinomial escrita en NumPy, con validación temporal. Muestra acierto o fallo. Resultado honesto: supera a la clase mayoritaria y a la persistencia en exactitud, pero empata con la persistencia en F1 macro |
| 1:55–2:30 | Chat: "¿Qué medidas de restricción vehicular se aplican en nivel de alerta?" | Parte 2: el agente decide usar `search_docs` (RAG sobre la Res. 2254 y el POECA) y cita [documento, artículo]. Abre el expander de la herramienta para mostrar los fragmentos |
| 2:30–3:00 | Chat: "El 15 de marzo de 2020, ¿la concentración en CEN-TRAF correspondía a algún nivel de episodio?" | La combinación: usa `get_pm25` (dato real) y `search_docs` (norma) en la misma respuesta |
| 3:00–3:20 | Chat: "¿Cómo está el aire en la estación de Narnia?" o una fecha de 2031 | Manejo de errores: el error de la herramienta vuelve al agente y se muestra un aviso, sin inventar datos |
| 3:20–3:45 | Terminal con `eval/resumen.json` o la tabla del README | Evaluación: 12 preguntas, 100 % herramienta correcta, citas presentes y válidas. Limitaciones: una sola corrida y sin datos en tiempo real. Cierre |

**Consejos:**

- Graba en 1080p con OBS o la grabación de pantalla de Windows (Win + Alt + R con la Xbox Game Bar).
- Si una respuesta del chat tarda, corta esa espera al editar.
- No muestres el archivo `.env` ni la consola de Groq: contienen tu clave.
