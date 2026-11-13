"""Agente con loop manual y límite de iteraciones T_MAX.

Cada iteración hace una llamada al LLM con las herramientas disponibles:
- Si el LLM pide herramientas, se ejecutan y su resultado (o su error) vuelve
  al LLM como observación, y empieza otra iteración.
- Si el LLM responde con texto, esa es la respuesta final.
Si se agotan las T_MAX iteraciones, se hace una última llamada SIN
herramientas para que responda con lo que ya observó.

Uso rápido desde la terminal:
    python -m src.agent "¿Cómo está el aire hoy en MED-ARAN?"
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from datetime import date

from src import llm as L
from src.tools import ESQUEMAS, ErrorHerramienta, Herramientas, es_error_herramienta  # noqa: F401

T_MAX = 5

# Algunos modelos (p. ej. gpt-oss) citan con corchetes lenticulares 【...】 y a veces dejan
# marcas internas como 【...predicted data...】. Se normaliza para que el usuario vea
# siempre [documento, ubicación] y el evaluador pueda reconocer las citas.
RE_LENTICULAR = re.compile(r"【([^】]*)】")
RE_DOC = re.compile(r"Res\.\s*2254|POECA|PIGECA|Informe SIATA")


def _citar(m: re.Match) -> str:
    dentro = m.group(1)
    interna = re.search(r"\[([^\[\]]+)\]", dentro)
    if interna and RE_DOC.search(interna.group(1)):
        return f"[{interna.group(1).strip()}]"
    if RE_DOC.search(dentro):
        limpio = dentro.strip().strip('"').strip()
        return f"[{limpio}]"
    return ""          # marca interna del modelo, no es una cita de documento


def limpiar_respuesta(texto: str) -> str:
    t = RE_LENTICULAR.sub(_citar, texto or "").replace("​", "")   # espacios de ancho cero
    return re.sub(r"[ \t  ]+([.,;:])", r"\1", t)   # espacio (también duro) antes de la puntuación


def instruccion_sistema(h: Herramientas, estacion_ui: str | None = None,
                        fecha_ui: str | None = None) -> str:
    grupos: dict[str, list[str]] = {}
    for e in h.estaciones():
        detalle = ", ".join(x for x in (e.get("nombre"), e.get("zona")) if x)
        etiqueta = (e["codigo"] + (f" ({detalle})" if detalle else "")
                    + ("" if e["activa"] else f" [sin datos desde {e['hasta']}]"))
        grupos.setdefault(e["municipio"] or "Sin municipio asignado", []).append(etiqueta)
    est = "; ".join(f"{m}: {', '.join(c)}" for m, c in grupos.items())
    extra = f"\nEl usuario tiene seleccionada la estación {estacion_ui} en el panel." if estacion_ui else ""
    if fecha_ui:
        extra += (f" Está viendo la fecha {fecha_ui}; úsala si pregunta por 'ese día' o no da fecha "
                  "(si dice 'hoy', usa el dato más reciente).")
    return f"""Eres un asistente de calidad del aire del Valle de Aburrá (Colombia). Respondes en español, de forma breve y clara.

Alcance (regla principal). Clasifica cada pregunta en uno de estos casos:
1. DENTRO DEL TEMA: cualquier pregunta que mencione el aire, la contaminación, las emisiones, el PM2.5, el ICA, la normativa ambiental o la salud ambiental del Valle de Aburrá (Medellín y sus municipios), AUNQUE la premisa sea rara o improbable (p. ej. "¿las plantas de mi casa limpian el aire de Medellín?" o "¿qué dice la norma sobre los olores de las curtiembres?"). Respóndela con las herramientas: busca con search_docs y responde solo con lo que digan los documentos. Si los documentos no respaldan la relación que plantea el usuario, dilo con claridad ("No encontré en los documentos evidencia de que…") y, si es útil, explica con citas cuáles son las fuentes o factores que sí documentan.
2. FUERA DE TEMA: preguntas sin relación con el aire ni con el ambiente (cultura general, personas, matemáticas, geografía de otros lugares, programación, etc.). NO las respondas, ni siquiera en parte, y no llames herramientas. Di con amabilidad que solo puedes ayudar con calidad del aire en el Valle de Aburrá y da dos o tres ejemplos de preguntas que sí puedes responder.
3. AMBIGUA: si no está claro a qué se refiere (p. ej. un nombre sin contexto), no adivines: pide que aclare su pregunta sobre calidad del aire.
Los saludos y agradecimientos se responden con normalidad, en una frase.

Reglas:
- Para datos medidos de PM2.5 usa la herramienta get_pm25. Para el pronóstico del día siguiente usa predict_tomorrow. No inventes cifras: toda cifra debe salir de una herramienta.
- Para preguntas sobre normas, niveles de prevención/alerta/emergencia, medidas, protocolos, el ICA o efectos en la salud usa search_docs. Responde SOLO con lo que digan los fragmentos y cita cada afirmación al final de la frase con el campo "cita" exacto, entre corchetes normales [ ], por ejemplo [POECA (Acuerdo Metropolitano 04 de 2018), Art. 27]. No uses 【 】 ni otras marcas de cita. No cites documentos que no aparezcan en los fragmentos.
- La "cita" indica a qué parte pertenece cada fragmento (por ejemplo "Art. 27, nivel de alerta" frente a "Art. 27, nivel de emergencia"). Si preguntan por un nivel, usa solo fragmentos de ese nivel o de las medidas comunes, y no mezcles medidas de niveles distintos. Lo mismo con las categorías del ICA: si preguntan por un color o categoría, usa solo lo que la tabla dice de esa categoría.
- No agregues efectos, recomendaciones ni datos que no estén en los fragmentos o en los resultados de las herramientas.
- Si los fragmentos no bastan, haz otra búsqueda con otras palabras (por ejemplo "medidas", "restricción", "declaratoria"). Si tras buscar no encuentras la respuesta, dilo explícitamente: "No encontré esto en los documentos consultados".
- Fecha de hoy: {date.today().isoformat()}. Los datos disponibles llegan hasta {h.fecha_max.date().isoformat()}; no hay datos en tiempo real. Si preguntan por "hoy", consulta sin fecha y explica qué fecha muestras.
- Si una herramienta devuelve un error, explícale al usuario qué pasó (por ejemplo, ofrece las estaciones candidatas) en vez de inventar.
- Si el usuario nombra un municipio (no un código) y la herramienta devuelve "otras_estaciones_activas_del_municipio", llama también a la herramienta para esas estaciones y reporta cada una por separado.
- predict_tomorrow pronostica la CLASE, no la concentración: no presentes el PM2.5 medido del último día como si fuera el pronóstico.
- Indica la fuente: las mediciones son del SIATA; el pronóstico es de un modelo de este proyecto (no del SIATA) que en prueba empata con la persistencia.
- Termina con: "Herramienta analítica, no es consejo médico."

Estaciones (código y municipio): {est}.{extra}"""


@dataclass
class Paso:
    iteracion: int
    herramienta: str
    argumentos: dict | str
    resultado: dict | None = None
    error: str | None = None


@dataclass
class RespuestaAgente:
    texto: str
    pasos: list[Paso] = field(default_factory=list)
    iteraciones: int = 0
    agoto_t_max: bool = False
    error_llm: str | None = None


def ejecutar_herramienta(h: Herramientas, nombre: str, argumentos_json: str, iteracion: int) -> Paso:
    """Nunca lanza excepciones: cualquier falla queda como error en el paso."""
    try:
        args = json.loads(argumentos_json or "{}")
        if not isinstance(args, dict):
            raise ValueError("los argumentos deben ser un objeto JSON")
    except (json.JSONDecodeError, ValueError) as e:
        return Paso(iteracion, nombre, argumentos_json, error=f"Argumentos inválidos ({e}).")
    try:
        return Paso(iteracion, nombre, args, resultado=h.ejecutar(nombre, args))
    except Exception as e:  # nunca se cae el agente
        if es_error_herramienta(e):
            return Paso(iteracion, nombre, args, error=str(e))
        return Paso(iteracion, nombre, args, error=f"Error interno en {nombre}: {e.__class__.__name__}: {e}")


def responder(pregunta: str, herramientas: Herramientas, cliente=None,
              historial: list[dict] | None = None, estacion_ui: str | None = None,
              fecha_ui: str | None = None,
              t_max: int = T_MAX) -> RespuestaAgente:
    """`historial`: turnos previos [{"role": "user"|"assistant", "content": ...}]."""
    cliente = cliente or L.ClienteLLM()
    mensajes = [{"role": "system", "content": instruccion_sistema(herramientas, estacion_ui, fecha_ui)}]
    mensajes += list(historial or [])
    mensajes.append({"role": "user", "content": pregunta})
    pasos: list[Paso] = []

    for it in range(1, t_max + 1):
        try:
            m = cliente.chat(mensajes, ESQUEMAS)
        except L.ErrorLLM as e:
            return RespuestaAgente(f"No pude contactar el modelo de lenguaje: {e}", pasos, it, error_llm=str(e))
        if not m.llamadas:
            return RespuestaAgente(limpiar_respuesta(m.contenido or ""), pasos, it)
        mensajes.append(L.mensaje_asistente(m))
        for c in m.llamadas:
            paso = ejecutar_herramienta(herramientas, c.nombre, c.argumentos, it)
            pasos.append(paso)
            obs = paso.resultado if paso.error is None else {"error": paso.error}
            mensajes.append(L.mensaje_herramienta(c.id, obs))

    # T_MAX agotado: una última llamada sin herramientas para cerrar.
    mensajes.append({"role": "user", "content": (
        "Se alcanzó el límite de consultas. Responde ahora con la información que ya tienes; "
        "si no alcanza, dilo.")})
    try:
        m = cliente.chat(mensajes, None)
        texto = limpiar_respuesta(m.contenido or "")
    except L.ErrorLLM as e:
        return RespuestaAgente(f"No pude contactar el modelo de lenguaje: {e}", pasos, t_max,
                               agoto_t_max=True, error_llm=str(e))
    return RespuestaAgente(texto, pasos, t_max, agoto_t_max=True)


def main(argv=None) -> None:
    pregunta = " ".join(argv or sys.argv[1:]) or "¿Cómo está el aire hoy en MED-ARAN?"
    r = responder(pregunta, Herramientas.desde_disco())
    for p in r.pasos:
        print(f"[iteración {p.iteracion}] {p.herramienta}({p.argumentos}) -> "
              + (f"ERROR: {p.error}" if p.error else json.dumps(p.resultado, ensure_ascii=False)[:200]))
    print(f"\n{r.texto}\n\n({r.iteraciones} iteraciones{', T_MAX agotado' if r.agoto_t_max else ''})")


if __name__ == "__main__":
    main(sys.argv[1:])
