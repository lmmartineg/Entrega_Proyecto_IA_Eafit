"""Genera informe.pdf a partir de docs/informe.md.

Uso (desde la raíz del proyecto):
    python scripts/generar_informe.py            # escribe informe.pdf
    python scripts/generar_informe.py --previsualizar   # además, PNG de cada página en docs/previa/

Soporta el subconjunto de Markdown que usa el informe: encabezado YAML simple
(titulo, subtitulo, autor), títulos ##, párrafos, listas con "-" o "1.", tablas
con barras, citas ">" y **negrita**, *cursiva* y `código` en línea.

Usa Arial (fuentes de Windows) para tener "≤", "−", "µ", etc.; si no está, usa
Helvetica y reemplaza los caracteres que esa fuente no tiene.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

RAIZ = Path(__file__).resolve().parents[1]
FUENTE_MD = RAIZ / "docs" / "informe.md"
SALIDA = RAIZ / "informe.pdf"
CUERPO, TABLA = 9.6, 8.4


def registrar_fuentes() -> tuple[str, str, str, str]:
    carpeta = Path("C:/Windows/Fonts")
    archivos = {"Arial": "arial.ttf", "Arial-Bold": "arialbd.ttf", "Arial-Italic": "ariali.ttf",
                "Arial-BoldItalic": "arialbi.ttf"}
    if all((carpeta / f).exists() for f in archivos.values()):
        for nombre, f in archivos.items():
            pdfmetrics.registerFont(TTFont(nombre, str(carpeta / f)))
        pdfmetrics.registerFontFamily("Arial", normal="Arial", bold="Arial-Bold",
                                      italic="Arial-Italic", boldItalic="Arial-BoldItalic")
        return "Arial", "Arial-Bold", "Arial-Italic", "Arial-BoldItalic"
    return "Helvetica", "Helvetica-Bold", "Helvetica-Oblique", "Helvetica-BoldOblique"


NORMAL, NEGRITA, CURSIVA, _ = registrar_fuentes()
SIN_GLIFOS = {"≤": "<=", "≥": ">=", "−": "-", "→": "->", "≈": "~"} if NORMAL == "Helvetica" else {}

AZUL = colors.HexColor("#1F3A5F")
estilos = {
    "titulo": ParagraphStyle("titulo", fontName=NEGRITA, fontSize=15, leading=18, alignment=TA_CENTER,
                             textColor=AZUL, spaceAfter=2),
    "subtitulo": ParagraphStyle("subtitulo", fontName=NORMAL, fontSize=9.5, leading=12, alignment=TA_CENTER,
                                textColor=colors.HexColor("#444444"), spaceAfter=1),
    "h2": ParagraphStyle("h2", fontName=NEGRITA, fontSize=11.5, leading=14, textColor=AZUL,
                         spaceBefore=7, spaceAfter=3),
    "p": ParagraphStyle("p", fontName=NORMAL, fontSize=CUERPO, leading=CUERPO * 1.28, spaceAfter=3.5),
    "item": ParagraphStyle("item", fontName=NORMAL, fontSize=CUERPO, leading=CUERPO * 1.25, leftIndent=12,
                           bulletIndent=3, spaceAfter=1.5),
    "cita": ParagraphStyle("cita", fontName=NORMAL, fontSize=CUERPO - 0.6, leading=(CUERPO - 0.6) * 1.3,
                           backColor=colors.HexColor("#EEF2F7"), borderPadding=(4, 6, 4, 6), spaceBefore=4,
                           spaceAfter=7, textColor=colors.HexColor("#222222")),
    "celda": ParagraphStyle("celda", fontName=NORMAL, fontSize=TABLA, leading=TABLA * 1.2),
    "celda_h": ParagraphStyle("celda_h", fontName=NEGRITA, fontSize=TABLA, leading=TABLA * 1.2,
                              textColor=colors.white),
}


def en_linea(texto: str) -> str:
    """Markdown en línea -> marcado de reportlab (escapa primero &, <, >)."""
    for a, b in SIN_GLIFOS.items():
        texto = texto.replace(a, b)
    t = texto.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    t = re.sub(r"`([^`]+)`", r'<font face="Courier" size="8.6">\1</font>', t)
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<i>\1</i>", t)
    t = re.sub(r"(https?://[^\s)]+)", r'<link href="\1" color="#1F3A5F">\1</link>', t)
    return t


def tabla(filas: list[str], ancho: float) -> Table:
    celdas = [[c.strip() for c in f.strip().strip("|").split("|")] for f in filas]
    celdas = [c for c in celdas if not all(re.fullmatch(r":?-{3,}:?", x) for x in c)]
    n = len(celdas[0])
    largos = [max(len(re.sub(r"[*`]", "", f[i])) for f in celdas) for i in range(n)]
    pesos = [min(max(l, 6), 60) for l in largos]
    anchos = [ancho * p / sum(pesos) for p in pesos]
    datos = [[Paragraph(en_linea(c), estilos["celda_h" if i == 0 else "celda"]) for c in f]
             for i, f in enumerate(celdas)]
    t = Table(datos, colWidths=anchos, repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), AZUL),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F3F5F8")]),
        ("LINEBELOW", (0, 0), (-1, -1), 0.3, colors.HexColor("#C9D1DC")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


def leer(md: str) -> tuple[dict, list[str]]:
    meta = {}
    lineas = md.splitlines()
    if lineas and lineas[0].strip() == "---":
        fin = lineas.index("---", 1)
        for l in lineas[1:fin]:
            k, _, v = l.partition(":")
            meta[k.strip()] = v.strip().strip('"')
        lineas = lineas[fin + 1:]
    return meta, lineas


def construir(md: str, ancho: float) -> list:
    meta, lineas = leer(md)
    historia = [Paragraph(en_linea(meta.get("titulo", "Informe")), estilos["titulo"])]
    for k in ("subtitulo", "autor"):
        if meta.get(k):
            historia.append(Paragraph(en_linea(meta[k]), estilos["subtitulo"]))
    historia.append(Spacer(1, 6))
    parrafo: list[str] = []

    def cerrar():
        if parrafo:
            historia.append(Paragraph(en_linea(" ".join(parrafo)), estilos["p"]))
            parrafo.clear()

    i = 0
    while i < len(lineas):
        l = lineas[i].rstrip()
        if not l.strip():
            cerrar()
        elif l.startswith("## "):
            cerrar()
            historia.append(Paragraph(en_linea(l[3:]), estilos["h2"]))
        elif l.startswith("|"):
            cerrar()
            bloque = []
            while i < len(lineas) and lineas[i].startswith("|"):
                bloque.append(lineas[i])
                i += 1
            historia += [tabla(bloque, ancho), Spacer(1, 5)]
            continue
        elif l.startswith("> "):
            cerrar()
            historia.append(Paragraph(en_linea(l[2:]), estilos["cita"]))
        elif re.match(r"^- ", l):
            cerrar()
            historia.append(Paragraph(en_linea(l[2:]), estilos["item"], bulletText="•"))
        elif re.match(r"^\d+\. ", l):
            cerrar()
            num, _, resto = l.partition(". ")
            historia.append(Paragraph(en_linea(resto), estilos["item"], bulletText=f"{num}."))
        else:
            parrafo.append(l.strip())
        i += 1
    cerrar()
    return historia


def pie(canvas, doc):
    canvas.saveState()
    canvas.setFont(NORMAL, 7.5)
    canvas.setFillColor(colors.HexColor("#666666"))
    canvas.drawString(doc.leftMargin, 1.1 * cm,
                      "SI3003 · Calidad del aire en el Valle de Aburrá · herramienta analítica, no es consejo médico")
    canvas.drawRightString(doc.pagesize[0] - doc.rightMargin, 1.1 * cm, f"{doc.page}")
    canvas.restoreState()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--previsualizar", action="store_true", help="exporta cada página a PNG en docs/previa/")
    a = ap.parse_args(argv)
    margen = 1.8 * cm
    doc = SimpleDocTemplate(str(SALIDA), pagesize=letter, leftMargin=margen, rightMargin=margen,
                            topMargin=1.6 * cm, bottomMargin=1.7 * cm, title="Informe SI3003 - Calidad del aire",
                            author="Proyecto SI3003")
    doc.build(construir(FUENTE_MD.read_text(encoding="utf-8"), doc.width), onFirstPage=pie, onLaterPages=pie)
    from pypdf import PdfReader
    n = len(PdfReader(str(SALIDA)).pages)
    print(f"{SALIDA.name}: {n} páginas (fuente {NORMAL})" + ("  <-- supera el máximo de 4" if n > 4 else ""))
    if a.previsualizar:
        import pypdfium2 as pdfium
        carpeta = RAIZ / "docs" / "previa"
        carpeta.mkdir(exist_ok=True)
        pdf = pdfium.PdfDocument(str(SALIDA))
        for k in range(len(pdf)):
            pdf[k].render(scale=1.4).to_pil().save(carpeta / f"pagina_{k + 1}.png")
        print(f"Vista previa en {carpeta}")


if __name__ == "__main__":
    main(sys.argv[1:])
