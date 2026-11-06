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
import sys
from dataclasses import dataclass, field
from datetime import date

from src import llm as L
from src.tools import ESQUEMAS, ErrorHerramienta, Herramientas

T_MAX = 5


def instruccion_sistema(h: Herramientas, estacion_ui: str | None = None) -> str:
    grupos: dict[str, list[str]] = {}
    for e in h.estaciones():
        etiqueta = e["codigo"] + ("" if e["activa"] else f" [sin datos desde {e['hasta']}]")
        grupos.setdefault(e["municipio"] or "Sin municipio asignado", []).append(etiqueta)
    est = "; ".join(f"{m}: {', '.join(c)}" for m, c in grupos.items())
    extra = f"\nEl usuario tiene seleccionada la estación {estacion_ui} en el panel." if estacion_ui else ""
    return f"""Eres un asistente de calidad del aire del Valle de Aburrá (Colombia). Respondes en español, de forma breve y clara.

Reglas:
- Para datos medidos de PM2.5 usa la herramienta get_pm25. Para el pronóstico del día siguiente usa predict_tomorrow. No inventes cifras: toda cifra debe salir de una herramienta.
- Para preguntas sobre normas, niveles de prevención/alerta/emergencia, medidas, protocolos, el ICA o efectos en la salud usa search_docs. Responde SOLO con lo que digan los fragmentos y cita cada afirmación al final de la frase con el campo "cita" exacto, por ejemplo [POECA (Acuerdo Metropolitano 04 de 2018), Art. 27]. No cites documentos que no aparezcan en los fragmentos.
- La "cita" indica a qué parte pertenece cada fragmento (por ejemplo "Art. 27, nivel de alerta" frente a "Art. 27, nivel de emergencia"). Si preguntan por un nivel, usa solo fragmentos de ese nivel o de las medidas comunes, y no mezcles medidas de niveles distintos.
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
    except ErrorHerramienta as e:
        return Paso(iteracion, nombre, args, error=str(e))
    except Exception as e:  # falla inesperada: se informa, no se cae el agente
        return Paso(iteracion, nombre, args, error=f"Error interno en {nombre}: {e.__class__.__name__}: {e}")


def responder(pregunta: str, herramientas: Herramientas, cliente=None,
              historial: list[dict] | None = None, estacion_ui: str | None = None,
              t_max: int = T_MAX) -> RespuestaAgente:
    """`historial`: turnos previos [{"role": "user"|"assistant", "content": ...}]."""
    cliente = cliente or L.ClienteLLM()
    mensajes = [{"role": "system", "content": instruccion_sistema(herramientas, estacion_ui)}]
    mensajes += list(historial or [])
    mensajes.append({"role": "user", "content": pregunta})
    pasos: list[Paso] = []

    for it in range(1, t_max + 1):
        try:
            m = cliente.chat(mensajes, ESQUEMAS)
        except L.ErrorLLM as e:
            return RespuestaAgente(f"No pude contactar el modelo de lenguaje: {e}", pasos, it, error_llm=str(e))
        if not m.llamadas:
            return RespuestaAgente(m.contenido or "", pasos, it)
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
        texto = m.contenido or ""
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
