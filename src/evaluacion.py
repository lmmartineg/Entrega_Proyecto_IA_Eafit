"""Evaluación del agente: corre preguntas fijas y mide herramienta, citas y contenido.

La ejecuta `python eval/run_eval.py`; aquí está la lógica para poder probarla.

Métricas por pregunta:
- herramienta_correcta: se llamaron todas las herramientas esperadas.
- cita_presente: la respuesta cita [documento, ubicación] (solo si la pregunta lo exige).
- citas_validas: cada cita de la respuesta coincide con la "cita" de un fragmento que
  devolvió search_docs en esa misma conversación (es decir, no es inventada).
- contenido_ok: pasa las comprobaciones de la pregunta (valor real de PM2.5, clase que
  predice el modelo y expresiones de `debe_contener`).
- aviso_medico: incluye "no es consejo médico".
"""
from __future__ import annotations

import re
import time
import unicodedata

import pandas as pd

from src import agent as A
from src import features as F

RE_CITA = re.compile(r"\[([^\[\]]*?(?:Res\.\s*2254|POECA|PIGECA|Informe SIATA)[^\[\]]*)\]")
GUIONES = re.compile(r"[‐-―−]")
PALABRAS_CLASE = {
    F.CLASES[0]: r"\bbuena\b",
    F.CLASES[1]: r"\baceptable\b",
    F.CLASES[2]: r"dañina|naranja|por encima|supera",
}


def normalizar(texto: str) -> str:
    """Unifica guiones raros (el LLM usa U+2011), espacios y forma Unicode."""
    t = unicodedata.normalize("NFC", texto)
    t = GUIONES.sub("-", t).replace(" ", " ")
    return re.sub(r"\s+", " ", t).strip()


def citas_en(texto: str) -> list[str]:
    return [normalizar(c) for c in RE_CITA.findall(texto or "")]


def patron_valor(valor: float) -> str:
    """17.0 -> acepta "17", "17.0" o "17,0"; 50.2 -> acepta "50.2" o "50,2"."""
    entero, dec = f"{valor:.1f}".split(".")
    if dec == "0":
        return rf"(?<![\d.,]){entero}(?:[.,]0)?(?![.,]?\d)"
    return rf"(?<![\d.,]){entero}[.,]{dec}(?![.,]?\d)"


def comprobar_contenido(item: dict, texto: str, herramientas) -> tuple[bool | None, list[str]]:
    """Devuelve (ok, detalles). ok=None si la pregunta no tiene comprobaciones."""
    detalles, oks = [], []
    t = normalizar(texto or "").lower()
    v = item.get("verificar")
    if v:
        if v["tipo"] == "pm25":
            real = herramientas.get_pm25(v["estacion"], v.get("fecha"))["pm25_promedio_diario_ug_m3"]
            ok = re.search(patron_valor(real), t) is not None
            detalles.append(f"PM2.5 real {real} {'encontrado' if ok else 'NO encontrado'}")
        elif v["tipo"] == "prediccion":
            clase = herramientas.predict_tomorrow(v["estacion"], v.get("fecha"))["clase_predicha"]
            ok = re.search(PALABRAS_CLASE[clase], t) is not None
            detalles.append(f"clase del modelo '{clase}' {'nombrada' if ok else 'NO nombrada'}")
        else:
            raise ValueError(f"Tipo de verificación desconocido: {v['tipo']}")
        oks.append(ok)
    for patron in item.get("debe_contener") or []:
        ok = re.search(patron, t, flags=re.IGNORECASE) is not None
        detalles.append(f"/{patron}/ {'sí' if ok else 'NO'}")
        oks.append(ok)
    for patron in item.get("no_debe_contener") or []:
        ok = re.search(patron, t, flags=re.IGNORECASE) is None
        detalles.append(f"sin /{patron}/ {'sí' if ok else 'NO (aparece)'}")
        oks.append(ok)
    return (all(oks) if oks else None), detalles


def citas_de_herramientas(pasos) -> set[str]:
    """Referencias que entregan get_pm25 y predict_tomorrow en su campo "norma", p. ej.
    "Res. 2254 de 2017 (límite diario: Art. 2; ICA: Tabla 6; niveles: Tabla 4)" ->
    {[Res. 2254 de 2017, Art. 2], [Res. 2254 de 2017, Tabla 6], [Res. 2254 de 2017, Tabla 4]}.
    Citar esas referencias no es inventar: vienen del resultado de una herramienta."""
    out = set()
    for p in pasos:
        norma = (p.resultado or {}).get("norma") if isinstance(p.resultado, dict) else None
        if not norma:
            continue
        doc = norma.split("(")[0].strip()
        for ref in re.findall(r"(Art\.\s*\d+|Tabla\s*\d+)", norma):
            out.add(normalizar(f"[{doc}, {ref}]"))
    return out


def evaluar_pregunta(item: dict, herramientas, cliente, t_max: int = A.T_MAX) -> dict:
    t0 = time.perf_counter()
    r = A.responder(item["pregunta"], herramientas, cliente, t_max=t_max)
    segundos = time.perf_counter() - t0
    llamadas = list(dict.fromkeys(p.herramienta for p in r.pasos))
    esperadas = item.get("herramientas_esperadas") or []
    citas = citas_en(r.texto)
    devueltas = {normalizar(f["cita"]) for p in r.pasos
                 if p.herramienta == "search_docs" and p.resultado
                 for f in p.resultado.get("fragmentos", [])}
    devueltas |= citas_de_herramientas(r.pasos)
    invalidas = [c for c in citas if f"[{c}]" not in devueltas]
    contenido_ok, detalles = comprobar_contenido(item, r.texto, herramientas)
    return {
        "id": item["id"], "categoria": item.get("categoria", ""), "pregunta": item["pregunta"],
        "herramientas_esperadas": ",".join(esperadas), "herramientas_llamadas": ",".join(llamadas),
        # sin_herramientas: la respuesta correcta es NO llamar ninguna (pregunta fuera de tema).
        "herramienta_correcta": (not llamadas) if item.get("sin_herramientas") else set(esperadas) <= set(llamadas),
        "herramientas_extra": ",".join(sorted(set(llamadas) - set(esperadas))),
        "requiere_cita": bool(item.get("requiere_cita")),
        "cita_presente": bool(citas), "n_citas": len(citas),
        "citas_validas": (not invalidas) if citas else None,
        "citas_invalidas": " | ".join(invalidas),
        "contenido_ok": contenido_ok, "detalle_contenido": "; ".join(detalles),
        "aviso_medico": "no es consejo médico" in normalizar(r.texto).lower(),
        "iteraciones": r.iteraciones, "agoto_t_max": r.agoto_t_max,
        "errores_herramientas": sum(1 for p in r.pasos if p.error),
        # El mensaje de error de la API trae el id de la organización: no se guarda.
        "error_llm": re.sub(r"org_\w+", "org_<oculto>", r.error_llm or ""),
        "segundos": round(segundos, 1), "respuesta": re.sub(r"org_\w+", "org_<oculto>", r.texto),
    }


def _pct(serie: pd.Series) -> float | None:
    s = serie.dropna()
    return round(100 * s.astype(bool).mean(), 1) if len(s) else None


def resumen(filas: pd.DataFrame) -> dict:
    con_cita = filas[filas["requiere_cita"]]
    return {
        "n_preguntas": int(len(filas)),
        "herramienta_correcta_pct": _pct(filas["herramienta_correcta"]),
        "cita_presente_pct (preguntas que la exigen)": _pct(con_cita["cita_presente"]),
        "citas_validas_pct (respuestas con citas)": _pct(filas["citas_validas"]),
        "contenido_ok_pct (preguntas con comprobación)": _pct(filas["contenido_ok"]),
        "aviso_medico_pct": _pct(filas["aviso_medico"]),
        "errores_llm": int(filas["error_llm"].fillna("").astype(str).str.strip().ne("").sum()),
        "segundos_mediana": float(filas["segundos"].median()) if len(filas) else None,
    }


def _marca(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "—"
    return "✅" if bool(v) else "❌"


def informe_markdown(filas: pd.DataFrame, res: dict) -> str:
    """eval/RESULTADOS.md: tabla legible de las preguntas y las respuestas completas."""
    nombres = {"herramienta_correcta_pct": "Herramienta correcta",
               "cita_presente_pct (preguntas que la exigen)": "Cita presente (en las que la exigen)",
               "citas_validas_pct (respuestas con citas)": "Citas válidas (no inventadas)",
               "contenido_ok_pct (preguntas con comprobación)": "Contenido verificado",
               "aviso_medico_pct": "Aviso médico"}
    lineas = [
        "# Evaluación del agente: resultados por pregunta", "",
        f"Corrida **{res.get('etiqueta', '')}** del {str(res.get('fecha', ''))[:10]} · LLM: "
        f"{res.get('llm', '')} (`{res.get('modelo', '')}`) · {len(filas)} preguntas fijas "
        "(`eval/preguntas.yaml`). Se genera con `python eval/run_eval.py`.", "",
        "| Métrica | Resultado |", "| --- | --- |",
        *[f"| {n} | {res[k]:.0f} % |" for k, n in nombres.items() if res.get(k) is not None], "",
        "**Cómo leer la tabla.** *Herramienta*: se llamaron las herramientas esperadas. *Cita*: la respuesta "
        "cita [documento, ubicación] (solo se exige en normativa). *Válidas*: cada cita corresponde a un "
        "fragmento que devolvió `search_docs`. *Contenido*: el valor de PM2.5 coincide con el dato real, se "
        "nombra la clase del modelo o aparecen las frases esperadas.", "",
        "| ID | Categoría | Pregunta | Esperada | Llamadas | Herramienta | Cita | Válidas | Contenido |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, f in filas.iterrows():
        cita = _marca(f["cita_presente"]) if f["requiere_cita"] else "no se exige"
        lineas.append(
            f"| {f['id']} | {f['categoria']} | {f['pregunta']} | `{f['herramientas_esperadas'] or '—'}` | "
            f"`{f['herramientas_llamadas'] or 'ninguna'}` | {_marca(f['herramienta_correcta'])} | {cita} | "
            f"{_marca(f['citas_validas'])} | {_marca(f['contenido_ok'])} |")
    lineas += ["", "## Respuestas del agente", ""]
    for _, f in filas.iterrows():
        resp = str(f["respuesta"]).strip().replace("\n", "\n> ")
        lineas += [f"<details><summary><b>{f['id']}</b> · {f['pregunta']}</summary>", "",
                   f"*Herramientas:* `{f['herramientas_llamadas'] or 'ninguna'}` · *comprobación:* "
                   f"{f['detalle_contenido'] or '—'}", "", f"> {resp}", "", "</details>", ""]
    return "\n".join(lineas)


def por_categoria(filas: pd.DataFrame) -> pd.DataFrame:
    return (filas.groupby("categoria")
            .agg(n=("id", "size"), herramienta_correcta=("herramienta_correcta", "mean"),
                 contenido_ok=("contenido_ok", lambda s: s.dropna().astype(bool).mean() if s.notna().any() else None))
            .round(2).reset_index())
