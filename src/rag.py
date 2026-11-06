"""RAG sobre la normativa: fragmentos con fuente, embeddings y un índice FAISS en disco.

Uso (desde la raíz del proyecto):
    python -m src.rag construir                          # lee data/corpus/ y escribe index/
    python -m src.rag buscar "¿qué pasa en nivel de alerta?"

Fuentes y unidad de cita:
- HTML (Res. 2254, POECA, PIGECA): se cita por artículo ("Art. 18"); el
  documento técnico del PIGECA, por sección numerada ("Secc. 2.1").
- PDF (informe del SIATA): se cita por página ("pág. 3").
- Las Tablas 4, 5 y 6 de la Res. 2254 son imágenes en la fuente; se agregan
  como un documento aparte con la transcripción de src/ica.py.

Embeddings: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 con
fastembed (ONNX, sin PyTorch; ~0,22 GB que se descargan la primera vez a
.cache/). Ese modelo lee unos 128 tokens, por eso los fragmentos son cortos
(~110 palabras, con 30 de solape) y llevan delante su documento y artículo.
Índice: FAISS IndexFlatIP sobre vectores normalizados (similitud coseno).

Búsqueda híbrida: el modelo de embeddings es pequeño y a veces no recupera el
artículo exacto (p. ej. "límite máximo permisible" -> Art. 2). Por eso se
combina con BM25 (coincidencia de palabras, implementado aquí) mediante fusión
por rango recíproco (RRF): puntaje = sum 1 / (60 + rango) en cada lista.
"""
from __future__ import annotations

import argparse
import html
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from src import config as cfg
from src import ica

MODELO_EMB = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MAX_PALABRAS = 110
SOLAPE = 30
K_DEFECTO = 8   # con 6, 2 de las 12 consultas de ajuste quedaban en los puestos 7-8

# Consultas con la cita esperada, para `python -m src.rag evaluar`. OJO: se usaron
# para ajustar la búsqueda (plurales, k, regla del mejor resultado de cada método),
# así que su resultado es optimista; la evaluación independiente es la de la Sesión 6.
_P, _R = "POECA (Acuerdo Metropolitano 04 de 2018)", "Res. 2254 de 2017"
_T = "Res. 2254 de 2017, Tablas 4-6 (transcripción del proyecto)"
CONSULTAS_AJUSTE = [
    ("¿Qué pasa si se declara nivel de alerta?", [f"{_P}, Art. 27", f"{_P}, Art. 18", f"{_T}, Tabla 4", f"{_R}, Art. 10"]),
    ("¿Cuál es el rango de concentración de PM2.5 para el nivel de alerta?", [f"{_T}, Tabla 4"]),
    ("¿Cuál es el límite máximo permisible de PM2.5 en 24 horas?", [f"{_R}, Art. 2]"]),
    ("¿Qué medidas de restricción vehicular se aplican en un episodio de contaminación?", [f"{_P}, Art. 27"]),
    ("¿Qué efectos en la salud tiene la categoría naranja del ICA?", [f"{_T}, Tabla 5"]),
    ("¿Quién declara los niveles de prevención, alerta o emergencia en el Valle de Aburrá?", [f"{_P}, Art. 18", f"{_R}, Art. 9]"]),
    ("¿Cuándo es el periodo de gestión de episodios de contaminación?", [f"{_P}, Art. 13"]),
    ("¿Qué porcentaje de datos válidos se exige para calcular el promedio diario?", ["Informe SIATA enero 2025"]),
    ("¿Cuál es la meta del PIGECA a 2030?", ["PIGECA"]),
    ("¿Cómo se calcula el índice de calidad del aire?", [f"{_R}, Art. 21", f"{_R}, Art. 18"]),
    ("¿Qué deben hacer los grupos sensibles cuando el aire está en naranja?", [f"{_T}, Tabla 5"]),
    ("¿Qué se considera un episodio de contaminación atmosférica?", [f"{_P}, Art. 3]", f"{_P}, Art. 5]", f"{_P}, Art. 7]"]),
]

# Recorte del contenido útil de cada HTML (lo demás es navegación del sitio).
HTML = {
    "res_2254_2017_mads.html": {
        "inicio": r"^RESOLUCIÓN 2254 DE 2017", "fin": r"^Gestor Normativo de la Comisión",
        "secciones_numeradas": False},
    "poeca_acuerdo_04_2018_amva.html": {
        "inicio": r"^ACUERDO METROPOLITANO", "fin": r"^Compilación Jurídica del Área Metropolitana",
        "secciones_numeradas": False,
        # El Art. 27 (medidas, ~3.600 palabras) tiene una subsección por nivel. Sin
        # esto, un fragmento de "emergencia" podía citarse como si fuera de "alerta".
        "subsecciones": {
            r"^Medidas comunes para los niveles": "medidas comunes a los tres niveles",
            r"^NIVEL (DE )?PREVENCIÓN": "nivel de prevención",
            r"^NIVEL (DE )?ALERTA": "nivel de alerta",
            r"^NIVEL (DE )?EMERGENCIA": "nivel de emergencia",
        }},
    "pigeca_acuerdo_16_2017_amva.html": {
        "inicio": r"^ACUERDO METROPOLITANO", "fin": r"^Compilación Jurídica del Área Metropolitana",
        "secciones_numeradas": True,
        # Después del acuerdo viene el documento técnico del plan.
        "cortes": {r"^DOCUMENTO DEL PLAN INTEGRAL DE GESTIÓN DE LA CALIDAD DEL AIRE": "Documento técnico"}},
}
DOC_TABLAS = {"archivo": "(transcripción de src/ica.py)",
              "corto": "Res. 2254 de 2017, Tablas 4-6 (transcripción del proyecto)",
              "url": "https://normas.cra.gov.co/gestor/docs/resolucion_minambienteds_2254_2017.htm"}

RE_ART = re.compile(r"^ART[ÍI]CULO\s+(\d+)\s*[oº°]?\s*\.?\s*(.*)$")
RE_CTX = re.compile(r"^(CAP[ÍI]TULO|T[ÍI]TULO)\s+[IVXLCDM\d]+\b\.?", re.IGNORECASE)
# Sección numerada: "2. EMISIONES A LA ATMÓSFERA" (un nivel, en mayúsculas) o
# "2.1.1.1.1 Antecedentes" (varios niveles, título corto). "1. La movilidad..."
# (un nivel en minúsculas) es un ítem de lista, no una sección.
RE_NUM = re.compile(r"^(\d+)\.?\s+([A-ZÁÉÍÓÚÑÜ][A-ZÁÉÍÓÚÑÜ0-9 ,;:()/–\-]{3,})$")
RE_SUBNUM = re.compile(r"^(\d+(?:\.\d+)+)\.?\s+([A-ZÁÉÍÓÚÑÜ][^.]{2,80})$")
RUIDO = {"Concordancias", "Resumen de Notas de Vigencia", "Jurisprudencia Vigencia", "Legislación Anterior"}


# --- Extracción ------------------------------------------------------------------
def lineas_html(texto_html: str) -> list[str]:
    t = re.sub(r"(?is)<(script|style|head).*?</\1>", " ", texto_html)
    t = re.sub(r"(?i)<br\s*/?>|</(p|div|tr|li|h\d|table)>", "\n", t)
    t = html.unescape(re.sub(r"<[^>]+>", " ", t))
    # La página de la CRA codifica mal el símbolo µ: "37 íg/m3" significa "37 µg/m³".
    t = t.replace("íg/m3", "µg/m³")
    return [re.sub(r"[ \t\xa0]+", " ", l).strip() for l in t.splitlines() if l.strip()]


def recortar(lineas: list[str], inicio: str, fin: str) -> list[str]:
    i = next((k for k, l in enumerate(lineas) if re.search(inicio, l)), None)
    if i is None:
        raise ValueError(f"No encontré el inicio del contenido ({inicio}).")
    j = next((k for k in range(i + 1, len(lineas)) if re.search(fin, lineas[k])), len(lineas))
    return lineas[i:j]


def _titulo_mayusculas(resto: str) -> str:
    """'RANGOS DE ... EMERGENCIA. Los rangos...' -> 'RANGOS DE ... EMERGENCIA'."""
    m = re.match(r"^([^a-záéíóúñ]+?)\.(\s|$)", resto)
    return m.group(1).strip() if m and len(m.group(1)) > 3 else ""


def secciones(lineas: list[str], secciones_numeradas: bool = False,
              cortes: dict[str, str] | None = None,
              subsecciones: dict[str, str] | None = None) -> list[dict]:
    """Agrupa las líneas por artículo (y, si se pide, por sección numerada).
    `cortes`: {regex: etiqueta} para abrir una sección sin número en una línea dada.
    `subsecciones`: {regex: sufijo} para partir un artículo, p. ej. "Art. 27, nivel de alerta"."""
    out = [{"ubicacion": "Encabezado", "titulo": "", "lineas": []}]
    en_documento_tecnico = False
    articulo = None                          # (ubicación, título) del artículo en curso
    for l in lineas:
        if l in RUIDO or RE_CTX.match(l):
            continue
        corte = next((et for rx, et in (cortes or {}).items() if re.search(rx, l)), None)
        sub = next((et for rx, et in (subsecciones or {}).items() if re.search(rx, l)), None)
        m = None if en_documento_tecnico else RE_ART.match(l)
        n = (RE_NUM.match(l) or RE_SUBNUM.match(l)) if secciones_numeradas else None
        if corte:
            en_documento_tecnico = True
            articulo = None
            out.append({"ubicacion": corte, "titulo": l, "lineas": []})
        elif sub and articulo:
            out.append({"ubicacion": f"{articulo[0]}, {sub}", "titulo": articulo[1], "lineas": [l]})
        elif m:
            articulo = (f"Art. {int(m.group(1))}", _titulo_mayusculas(m.group(2)))
            out.append({"ubicacion": articulo[0], "titulo": articulo[1], "lineas": [l]})
        elif n:
            out.append({"ubicacion": f"Secc. {n.group(1)}", "titulo": n.group(2).strip(), "lineas": []})
        else:
            out[-1]["lineas"].append(l)
    return [{"ubicacion": s["ubicacion"], "titulo": s["titulo"], "texto": " ".join(s["lineas"])}
            for s in out if s["lineas"]]


def fragmentar(texto: str, max_palabras: int = MAX_PALABRAS, solape: int = SOLAPE) -> list[str]:
    palabras = texto.split()
    if len(palabras) <= max_palabras:
        return [" ".join(palabras)] if palabras else []
    paso = max_palabras - solape
    trozos = []
    for i in range(0, len(palabras), paso):
        trozos.append(" ".join(palabras[i:i + max_palabras]))
        if i + max_palabras >= len(palabras):
            break
    return trozos


def secciones_tablas() -> list[dict]:
    t4 = "; ".join(f"{n}: {lo:g} a {hi:g} µg/m³" if hi != float("inf") else f"{n}: {lo:g} µg/m³ o más"
                   for lo, hi, n in ica.NIVELES_PM25)
    t5 = " ".join(f"ICA {r} ({c}, {e}): {ef}" for r, c, e, ef in ica.EFECTOS_ICA)
    rangos_ica = ["0-50", "51-100", "101-150", "151-200", "201-300", "301-500"]
    t6 = "; ".join(f"ICA {r}, {c}, {n}: PM2.5 de {lo:g} a {hi:g} µg/m³"
                   for r, (lo, hi, n, c) in zip(rangos_ica, ica.ICA_PM25))
    return [
        {"ubicacion": "Tabla 4", "titulo": "CONCENTRACIONES PARA LOS NIVELES DE PREVENCIÓN, ALERTA O EMERGENCIA",
         "texto": f"Tabla 4 (Art. 10) de la Res. 2254 de 2017, PM2.5 promedio de 24 horas: {t4}. "
                  "Las declaraciones de niveles por PM2.5 aplican desde el 1 de julio de 2018."},
        {"ubicacion": "Tabla 5", "titulo": "DESCRIPCIÓN GENERAL DEL ÍNDICE DE CALIDAD DEL AIRE",
         "texto": f"Tabla 5 (Art. 19) de la Res. 2254 de 2017, efectos en la salud por rango del ICA. {t5}"},
        {"ubicacion": "Tabla 6", "titulo": "PUNTOS DE CORTE DEL ICA PARA PM2.5",
         "texto": f"Tabla 6 (Art. 20) de la Res. 2254 de 2017, puntos de corte del ICA para PM2.5 en 24 horas: {t6}."},
    ]


def construir_fragmentos(carpeta: Path | None = None) -> list[dict]:
    """Lee data/corpus/ y devuelve los fragmentos con su fuente."""
    from pypdf import PdfReader
    carpeta = Path(carpeta or cfg.DATA_CORPUS)
    fuentes = json.loads((carpeta / "_fuentes.json").read_text(encoding="utf-8"))["documentos"]
    frags = []

    def agregar(doc: dict, archivo: str, secs: list[dict]):
        for s in secs:
            for k, trozo in enumerate(fragmentar(s["texto"])):
                frags.append({
                    "id": len(frags), "documento": doc["corto"], "archivo": archivo, "url": doc["url"],
                    "ubicacion": s["ubicacion"], "titulo": s["titulo"], "parte": k + 1, "texto": trozo,
                })

    for doc in fuentes:
        ruta = carpeta / doc["archivo"]
        if doc["archivo"] in HTML:
            c = HTML[doc["archivo"]]
            lineas = recortar(lineas_html(ruta.read_bytes().decode("utf-8", errors="replace")),
                              c["inicio"], c["fin"])
            agregar(doc, doc["archivo"], secciones(lineas, c["secciones_numeradas"], c.get("cortes"),
                                                   c.get("subsecciones")))
        elif ruta.suffix.lower() == ".pdf":
            paginas = PdfReader(str(ruta)).pages
            secs = [{"ubicacion": f"pág. {i}", "titulo": "",
                     "texto": re.sub(r"\s+", " ", p.extract_text() or "")} for i, p in enumerate(paginas, 1)]
            agregar(doc, doc["archivo"], secs)
    agregar(DOC_TABLAS, DOC_TABLAS["archivo"], secciones_tablas())
    return frags


def texto_para_embeber(f: dict) -> str:
    cab = f"{f['documento']}, {f['ubicacion']}" + (f" {f['titulo']}" if f["titulo"] else "")
    return f"{cab}. {f['texto']}"


# --- Embeddings ---------------------------------------------------------------------
class Embebedor:
    """Envuelve fastembed. Devuelve vectores float32 normalizados (norma 1)."""

    def __init__(self, modelo: str = MODELO_EMB):
        from fastembed import TextEmbedding
        self.modelo = modelo
        self._m = TextEmbedding(modelo, cache_dir=str(cfg.CACHE / "fastembed"))

    def __call__(self, textos: list[str]) -> np.ndarray:
        v = np.asarray(list(self._m.embed(textos)), dtype="float32")
        return v / np.linalg.norm(v, axis=1, keepdims=True).clip(min=1e-12)


# --- BM25 (búsqueda por palabras) -----------------------------------------------------
STOPWORDS = set("""a al ante con contra de del desde durante e el en entre es esta este esto
hacia hasta la las lo los mas o para pero por que se sea segun ser si sin sobre su sus un una
uno unos y ya cual cuales cuando como donde quien quienes qué cuál cómo""".split())


def raiz(w: str) -> str:
    """Raíz mínima para que singular y plural coincidan: se quita la "s" final y
    luego la vocal final. niveles/nivel -> nivel; permisibles/permisible -> permisibl;
    maximos/maximo -> maxim; emisiones/emision -> emision. No es un lematizador."""
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        w = w[:-1]
    if len(w) > 4 and w[-1] in "aeo":
        w = w[:-1]
    return w


def tokens(texto: str) -> list[str]:
    import unicodedata
    t = unicodedata.normalize("NFKD", texto.lower()).encode("ascii", "ignore").decode()
    return [raiz(w) for w in re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", t)
            if w not in STOPWORDS and len(w) > 1]


class BM25:
    def __init__(self, documentos: list[str], k1: float = 1.5, b: float = 0.75):
        self.docs = [tokens(d) for d in documentos]
        self.k1, self.b = k1, b
        self.largo = np.array([len(d) for d in self.docs], dtype=float)
        self.largo_medio = self.largo.mean() if len(self.docs) else 0.0
        df: dict[str, int] = {}
        for d in self.docs:
            for w in set(d):
                df[w] = df.get(w, 0) + 1
        n = len(self.docs)
        self.idf = {w: np.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}
        self.tf = [{} for _ in self.docs]
        for i, d in enumerate(self.docs):
            for w in d:
                self.tf[i][w] = self.tf[i].get(w, 0) + 1

    def puntajes(self, consulta: str) -> np.ndarray:
        q = [w for w in tokens(consulta) if w in self.idf]
        s = np.zeros(len(self.docs))
        norm = self.k1 * (1 - self.b + self.b * self.largo / max(self.largo_medio, 1e-9))
        for w in set(q):
            f = np.array([tf.get(w, 0) for tf in self.tf], dtype=float)
            s += self.idf[w] * f * (self.k1 + 1) / (f + norm)
        return s


def fusion_rrf(rankings: list[list[int]], k_final: int, k: int = 60) -> list[tuple[int, float]]:
    """Fusión por rango recíproco. Además, el primer resultado de cada método
    siempre entra en los k_final: un acierto claro de un solo método (p. ej. BM25
    con "máximos permisibles") no se pierde por estar lejos en el otro."""
    puntaje: dict[int, float] = {}
    for r in rankings:
        for pos, i in enumerate(r):
            puntaje[i] = puntaje.get(i, 0.0) + 1.0 / (k + pos + 1)
    orden = sorted(puntaje.items(), key=lambda x: -x[1])
    top = orden[:k_final]
    for r in rankings:
        if r and r[0] not in {i for i, _ in top}:
            # Reemplaza el último que no sea el mejor de algún método.
            primeros = {rr[0] for rr in rankings if rr}
            j = max(idx for idx, (i, _) in enumerate(top) if i not in primeros)
            top[j] = (r[0], puntaje[r[0]])
    return sorted(top, key=lambda x: -x[1])


# --- Índice y búsqueda -------------------------------------------------------------
def construir_indice(fragmentos: list[dict], embebedor, carpeta: Path | None = None) -> Path:
    import faiss
    carpeta = Path(carpeta or cfg.INDEX)
    carpeta.mkdir(parents=True, exist_ok=True)
    vec = embebedor([texto_para_embeber(f) for f in fragmentos])
    indice = faiss.IndexFlatIP(vec.shape[1])
    indice.add(vec)
    faiss.write_index(indice, str(carpeta / "normativa.faiss"))
    with open(carpeta / "fragmentos.jsonl", "w", encoding="utf-8") as fh:
        for f in fragmentos:
            fh.write(json.dumps(f, ensure_ascii=False) + "\n")
    por_doc = {}
    for f in fragmentos:
        por_doc[f["documento"]] = por_doc.get(f["documento"], 0) + 1
    meta = {"modelo_embeddings": getattr(embebedor, "modelo", "desconocido"), "dimension": int(vec.shape[1]),
            "n_fragmentos": len(fragmentos), "max_palabras": MAX_PALABRAS, "solape": SOLAPE,
            "fragmentos_por_documento": por_doc, "construido": datetime.now().isoformat(timespec="seconds")}
    (carpeta / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return carpeta


class Buscador:
    def __init__(self, fragmentos: list[dict], indice, embebedor=None, modelo: str = MODELO_EMB):
        self.fragmentos = fragmentos
        self.indice = indice
        self._embebedor = embebedor
        self.modelo = modelo
        self.bm25 = BM25([texto_para_embeber(f) for f in fragmentos])

    @classmethod
    def cargar(cls, carpeta: Path | None = None, embebedor=None) -> "Buscador":
        import faiss
        carpeta = Path(carpeta or cfg.INDEX)
        if not (carpeta / "normativa.faiss").exists():
            raise FileNotFoundError(f"No hay índice en {carpeta}. Corre `python -m src.rag construir`.")
        meta = json.loads((carpeta / "meta.json").read_text(encoding="utf-8"))
        frags = [json.loads(l) for l in (carpeta / "fragmentos.jsonl").read_text(encoding="utf-8").splitlines()]
        return cls(frags, faiss.read_index(str(carpeta / "normativa.faiss")), embebedor,
                   meta.get("modelo_embeddings", MODELO_EMB))

    @property
    def embebedor(self):
        if self._embebedor is None:          # carga perezosa: el modelo pesa ~0,22 GB
            self._embebedor = Embebedor(self.modelo)
        return self._embebedor

    def buscar(self, consulta: str, k: int = K_DEFECTO, candidatos: int = 50,
               modo: str = "hibrido") -> list[dict]:
        """modo: "hibrido" (embeddings + BM25 con RRF), "denso" o "bm25"."""
        if not consulta or not consulta.strip():
            raise ValueError("La consulta está vacía.")
        n = min(candidatos, len(self.fragmentos))
        rankings, similitud = [], {}
        if modo in ("hibrido", "denso"):
            sim, ids = self.indice.search(self.embebedor([consulta]), n)
            rankings.append([int(i) for i in ids[0] if i >= 0])
            similitud = {int(i): float(s) for s, i in zip(sim[0], ids[0]) if i >= 0}
        if modo in ("hibrido", "bm25"):
            p = self.bm25.puntajes(consulta)
            orden = np.argsort(-p)[:n]
            rankings.append([int(i) for i in orden if p[i] > 0])
        out = []
        for i, puntaje in fusion_rrf(rankings, k):
            f = self.fragmentos[i]
            out.append({"cita": f"[{f['documento']}, {f['ubicacion']}]", "documento": f["documento"],
                        "ubicacion": f["ubicacion"], "titulo": f["titulo"], "texto": f["texto"],
                        "url": f["url"], "puntaje_rrf": round(puntaje, 4),
                        "similitud": round(similitud[i], 3) if i in similitud else None})
        return out


def evaluar(buscador: "Buscador", k: int = K_DEFECTO) -> list[dict]:
    filas = []
    for modo in ("denso", "bm25", "hibrido"):
        aciertos, rr, fallas = 0, 0.0, []
        for q, esperado in CONSULTAS_AJUSTE:
            r = buscador.buscar(q, k, modo=modo)
            pos = next((i for i, x in enumerate(r) if any(e in x["cita"] for e in esperado)), None)
            if pos is None:
                fallas.append(q)
            else:
                aciertos += 1
                rr += 1 / (pos + 1)
        filas.append({"modo": modo, "k": k, "aciertos": aciertos, "n": len(CONSULTAS_AJUSTE),
                      "mrr": round(rr / len(CONSULTAS_AJUSTE), 3), "fallas": fallas})
    return filas


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("construir")
    e = sub.add_parser("evaluar")
    e.add_argument("-k", type=int, default=K_DEFECTO)
    b = sub.add_parser("buscar")
    b.add_argument("consulta")
    b.add_argument("-k", type=int, default=K_DEFECTO)
    b.add_argument("--modo", choices=["hibrido", "denso", "bm25"], default="hibrido")
    a = ap.parse_args(argv)
    if a.cmd == "construir":
        frags = construir_fragmentos()
        print(f"{len(frags)} fragmentos. Calculando embeddings con {MODELO_EMB}...")
        carpeta = construir_indice(frags, Embebedor())
        print(json.dumps(json.loads((carpeta / "meta.json").read_text(encoding="utf-8")), indent=2,
                         ensure_ascii=False))
    elif a.cmd == "evaluar":
        filas = evaluar(Buscador.cargar(), a.k)
        for f in filas:
            print(f"{f['modo']:<8} aciertos@{f['k']} = {f['aciertos']}/{f['n']}  MRR = {f['mrr']:.2f}  "
                  f"fallas: {[q[:50] for q in f['fallas']]}")
        (cfg.INDEX / "evaluacion_recuperacion.json").write_text(
            json.dumps(filas, indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        for r in Buscador.cargar().buscar(a.consulta, a.k, modo=a.modo):
            print(f"{r['puntaje_rrf']:.4f}  {r['cita']}  {r['titulo'][:60]}\n       {r['texto'][:220]}...\n")


if __name__ == "__main__":
    main(sys.argv[1:])
