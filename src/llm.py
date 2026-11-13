"""Cliente LLM compatible con la API de OpenAI (Groq, NVIDIA NIM u Ollama).

La configuración se lee de `.env` (ver `.env.example`):

    LLM_PROVIDER=groq            # groq | nvidia | ollama
    LLM_MODEL=                   # opcional; si se deja vacío usa el del proveedor
    LLM_PROVIDER_ALT=ollama      # opcional; se usa si el principal falla
    GROQ_API_KEY=...             # la clave del proveedor que uses
    NVIDIA_API_KEY=...

El cliente devuelve mensajes en un formato propio y simple (`MensajeLLM`), así
el agente no depende del SDK y se puede probar con un LLM simulado.

Prueba de conexión (no imprime la clave):
    python -m src.llm            # lista modelos y prueba una llamada con herramientas
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

from src import config as cfg

# Modelos sugeridos: verifica que sigan disponibles en la lista de modelos de
# tu proveedor; los catálogos cambian con frecuencia.
PROVEEDORES = {
    # Verificado el 6 nov 2026 con `python -m src.llm`: llama-3.3-70b-versatile ya no
    # está en Groq; gpt-oss-120b, gpt-oss-20b y qwen3.8-27b sí, y los tres usan herramientas.
    "groq": {"base_url": "https://api.groq.com/openai/v1", "clave": "GROQ_API_KEY",
             "modelo": "openai/gpt-oss-120b"},
    "nvidia": {"base_url": "https://integrate.api.nvidia.com/v1", "clave": "NVIDIA_API_KEY",
               "modelo": "meta/llama-3.3-70b-instruct"},
    "ollama": {"base_url": "http://localhost:11434/v1", "clave": None,
               "modelo": "qwen2.5:7b"},
}


class ErrorLLM(RuntimeError):
    """Falla de la API del LLM (clave, red, límite de uso, modelo inexistente...)."""


ESPERA_MAXIMA = 30.0


def espera_sugerida(error: Exception, intento: int) -> float:
    """Segundos a esperar antes de reintentar. Groq indica el tiempo en el encabezado
    retry-after y en el mensaje ("Please try again in 10.42s"); si no lo hay, espera
    creciente 2, 4, 8 s. Se limita a ESPERA_MAXIMA."""
    import re
    s = None
    resp = getattr(error, "response", None)
    try:
        s = float(resp.headers.get("retry-after")) if resp is not None else None
    except (TypeError, ValueError, AttributeError):
        s = None
    if s is None:
        m = re.search(r"try again in ([\d.]+)\s*s", str(error))
        s = float(m.group(1)) if m else 2.0 * (2 ** intento)
    return min(s + 0.5, ESPERA_MAXIMA)


@dataclass
class LlamadaHerramienta:
    id: str
    nombre: str
    argumentos: str          # JSON en texto, tal como lo devuelve el modelo


@dataclass
class MensajeLLM:
    contenido: str | None
    llamadas: list[LlamadaHerramienta] = field(default_factory=list)
    proveedor: str = ""
    modelo: str = ""


def cargar_entorno() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv(cfg.ROOT / ".env")
    except ImportError:
        pass


class ClienteLLM:
    def __init__(self, proveedor: str | None = None, modelo: str | None = None,
                 proveedor_alt: str | None = None, temperatura: float = 0.0,
                 timeout: float = 60.0, reintentos: int = 3):
        self.reintentos = reintentos
        cargar_entorno()
        self.proveedor = (proveedor or os.getenv("LLM_PROVIDER") or "groq").lower()
        self.modelo = modelo or os.getenv("LLM_MODEL") or None
        alt = proveedor_alt if proveedor_alt is not None else os.getenv("LLM_PROVIDER_ALT", "")
        self.proveedor_alt = alt.lower() or None
        self.temperatura = temperatura
        self.timeout = timeout
        for p in filter(None, (self.proveedor, self.proveedor_alt)):
            if p not in PROVEEDORES:
                raise ErrorLLM(f"Proveedor desconocido: {p}. Opciones: {list(PROVEEDORES)}")

    def clave_faltante(self) -> str | None:
        """Mensaje para el usuario si falta la clave del proveedor principal; None si está."""
        conf = PROVEEDORES[self.proveedor]
        if conf["clave"] and not os.getenv(conf["clave"]):
            return (f"Falta {conf['clave']} en .env para usar {self.proveedor}. El chat necesita una clave "
                    "gratuita (https://console.groq.com/keys): copia .env.example como .env y pégala en "
                    f"{conf['clave']}= (ver README, «Inicio rápido»). El mapa, la serie y la predicción "
                    "funcionan sin clave.")
        return None

    def _cliente(self, proveedor: str):
        from openai import OpenAI
        conf = PROVEEDORES[proveedor]
        clave = os.getenv(conf["clave"]) if conf["clave"] else "ollama"
        if not clave:
            raise ErrorLLM(f"Falta {conf['clave']} en .env para usar {proveedor}. El chat necesita una clave "
                           "gratuita (https://console.groq.com/keys): copia .env.example como .env y pégala "
                           "(ver README, «Inicio rápido»). El mapa, la serie y la predicción funcionan sin clave.")
        return OpenAI(base_url=conf["base_url"], api_key=clave, timeout=self.timeout, max_retries=0)

    def _llamar(self, proveedor: str, modelo: str, mensajes: list[dict], herramientas):
        import openai
        cliente = self._cliente(proveedor)
        kwargs = {"model": modelo, "messages": mensajes, "temperature": self.temperatura}
        if herramientas:
            kwargs["tools"] = herramientas
        for intento in range(self.reintentos + 1):
            try:
                r = cliente.chat.completions.create(**kwargs)
                break
            except (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError,
                    openai.InternalServerError) as e:
                # El límite DIARIO no se recupera en segundos: reintentar solo hace esperar.
                if intento == self.reintentos or "per day" in str(e):
                    raise ErrorLLM(f"{proveedor}: {e.__class__.__name__}: {e}") from e
                time.sleep(espera_sugerida(e, intento))
            except openai.APIError as e:
                raise ErrorLLM(f"{proveedor}: {e.__class__.__name__}: {e}") from e
        m = r.choices[0].message
        llamadas = [LlamadaHerramienta(id=t.id, nombre=t.function.name,
                                       argumentos=t.function.arguments or "{}")
                    for t in (m.tool_calls or [])]
        return MensajeLLM(contenido=m.content, llamadas=llamadas, proveedor=proveedor, modelo=modelo)

    def chat(self, mensajes: list[dict], herramientas: list[dict] | None = None) -> MensajeLLM:
        """Una llamada al LLM. Si el proveedor principal falla y hay uno alterno, lo intenta."""
        intentos = [(self.proveedor, self.modelo or PROVEEDORES[self.proveedor]["modelo"])]
        if self.proveedor_alt and self.proveedor_alt != self.proveedor:
            intentos.append((self.proveedor_alt, PROVEEDORES[self.proveedor_alt]["modelo"]))
        errores = []
        for proveedor, modelo in intentos:
            try:
                return self._llamar(proveedor, modelo, mensajes, herramientas)
            except ErrorLLM as e:
                errores.append(str(e))
        raise ErrorLLM(" | ".join(errores))


def mensaje_asistente(m: MensajeLLM) -> dict:
    """Convierte la respuesta del LLM al formato de historial de la API."""
    d = {"role": "assistant", "content": m.contenido or ""}
    if m.llamadas:
        d["tool_calls"] = [{"id": c.id, "type": "function",
                            "function": {"name": c.nombre, "arguments": c.argumentos}}
                           for c in m.llamadas]
    return d


def mensaje_herramienta(id_llamada: str, resultado: dict) -> dict:
    return {"role": "tool", "tool_call_id": id_llamada,
            "content": json.dumps(resultado, ensure_ascii=False, default=str)}


def probar() -> None:
    """Diagnóstico de la configuración: clave, modelos disponibles y llamada con herramientas."""
    c = ClienteLLM(proveedor_alt="")
    conf = PROVEEDORES[c.proveedor]
    modelo = c.modelo or conf["modelo"]
    print(f"Proveedor: {c.proveedor} ({conf['base_url']})")
    if conf["clave"]:
        clave = os.getenv(conf["clave"]) or ""
        print(f"{conf['clave']}: {'presente (' + str(len(clave)) + ' caracteres)' if clave else 'FALTA'}")
        if not clave:
            raise SystemExit(f"Pega tu clave en .env, en la línea {conf['clave']}=")
    try:
        ids = sorted(m.id for m in c._cliente(c.proveedor).models.list())
    except Exception as e:
        raise SystemExit(f"No se pudo listar modelos: {e.__class__.__name__}: {e}")
    print(f"Modelos disponibles ({len(ids)}):")
    for i in ids:
        print(f"  {'*' if i == modelo else ' '} {i}")
    if modelo not in ids:
        print(f"\nAVISO: el modelo configurado '{modelo}' no está en la lista. "
              "Pon uno de la lista en LLM_MODEL (en .env).")
        return
    herramienta = [{"type": "function", "function": {
        "name": "sumar", "description": "Suma dos números",
        "parameters": {"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                       "required": ["a", "b"]}}}]
    m = c.chat([{"role": "user", "content": "Usa la herramienta sumar para calcular 2 + 3."}], herramienta)
    if m.llamadas:
        print(f"\nOK: {modelo} pidió la herramienta {m.llamadas[0].nombre}({m.llamadas[0].argumentos}).")
    else:
        print(f"\nAVISO: {modelo} respondió sin usar herramientas: {m.contenido!r}")


if __name__ == "__main__":
    probar()
