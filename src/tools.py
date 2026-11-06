"""Herramientas del agente: get_pm25, predict_tomorrow y search_docs.

Cada herramienta devuelve un dict serializable a JSON. Si algo falla por un
argumento del usuario (estación inexistente, fecha sin datos...), lanza
`ErrorHerramienta` con un mensaje que el agente recibe como observación.

Fechas: los datos descargados del SIATA llegan hasta la última fecha de
`pm25_diario.csv` (hoy, 2026-09-30). "hoy" o sin fecha significa "el dato más
reciente disponible", y la respuesta lo indica en `nota`.
"""
from __future__ import annotations

import unicodedata
from datetime import date

import numpy as np
import pandas as pd

from src import config as cfg
from src import features as F
from src import ica
from src.model import RegresionSoftmax

FUENTE_DATOS = ('SIATA, 2026, "Material Particulado - PM2.5", https://doi.org/10.83041/AUWZWT '
                "(promedio diario calculado por este proyecto)")

# Prefijo del código de estación -> municipio. Inferido de los 10 municipios del
# Valle de Aburrá; CEN y SUR no corresponden a un municipio y quedan sin asignar.
MUNICIPIOS = {"BAR": "Barbosa", "GIR": "Girardota", "COP": "Copacabana", "BEL": "Bello",
              "MED": "Medellín", "ITA": "Itagüí", "ENV": "Envigado", "SAB": "Sabaneta",
              "EST": "La Estrella", "CAL": "Caldas"}
PALABRAS_HOY = {"", "hoy", "ultimo", "último", "actual", "reciente", "ahora"}


class ErrorHerramienta(ValueError):
    """Error causado por los argumentos; el mensaje se le muestra al agente."""


def _norm(texto: str) -> str:
    t = unicodedata.normalize("NFKD", str(texto)).encode("ascii", "ignore").decode()
    return t.upper().strip()


class Herramientas:
    def __init__(self, diario: pd.DataFrame, modelo: RegresionSoftmax | None = None,
                 meta_modelo: dict | None = None, resumen_modelo: dict | None = None,
                 buscador=None):
        self.buscador = buscador          # src.rag.Buscador (None si no hay índice)
        self.diario = diario.copy()
        self.diario["fecha"] = pd.to_datetime(self.diario["fecha"])
        self.modelo = modelo
        self.meta_modelo = meta_modelo or {}
        self.resumen_modelo = resumen_modelo or {}
        self.codigos = sorted(self.diario["estacion"].unique())
        self.fecha_max = self.diario["fecha"].max()
        self._dataset = None

    @classmethod
    def desde_disco(cls) -> "Herramientas":
        diario = pd.read_csv(cfg.DATA_PROCESSED / "pm25_diario.csv", parse_dates=["fecha"])
        modelo, meta, resumen = None, {}, {}
        ruta = cfg.MODELS / "softmax_pm25"
        if ruta.with_suffix(".npz").exists():
            modelo, meta = RegresionSoftmax.cargar(ruta)
            ruta_m = cfg.MODELS / "metricas.csv"
            if ruta_m.exists():
                m = pd.read_csv(ruta_m)
                m = m[m["periodo"] == "test"].set_index("metodo")
                if {"softmax", "persistencia"} <= set(m.index):
                    resumen = {
                        "f1_macro_modelo": round(float(m.loc["softmax", "f1_macro"]), 3),
                        "f1_macro_persistencia": round(float(m.loc["persistencia", "f1_macro"]), 3),
                        "precision_alertas": round(float(m.loc["softmax", "precision_2"]), 3),
                        "sensibilidad_alertas": round(float(m.loc["softmax", "sensibilidad_2"]), 3),
                    }
        buscador = None
        if (cfg.INDEX / "normativa.faiss").exists():
            from src.rag import Buscador
            buscador = Buscador.cargar()
        return cls(diario, modelo, meta, resumen, buscador)

    # --- estaciones ------------------------------------------------------------
    def estaciones(self) -> list[dict]:
        g = self.diario[self.diario["valido"]].groupby("estacion")["fecha"].agg(["min", "max"])
        limite_activa = self.fecha_max - pd.Timedelta(days=30)
        return [{"codigo": c, "municipio": MUNICIPIOS.get(c.split("-")[0]),
                 "desde": g.loc[c, "min"].date().isoformat(), "hasta": g.loc[c, "max"].date().isoformat(),
                 "activa": bool(g.loc[c, "max"] >= limite_activa)}
                for c in g.index]

    def vecinas_activas(self, cod: str) -> list[str]:
        """Otras estaciones activas del mismo municipio (mismo prefijo de código)."""
        pref = cod.split("-")[0]
        if pref not in MUNICIPIOS:
            return []
        return [e["codigo"] for e in self.estaciones()
                if e["activa"] and e["codigo"] != cod and e["codigo"].startswith(pref + "-")]

    def resolver_estacion(self, texto: str) -> str:
        """Código exacto, código sin guion, municipio o fragmento del código."""
        if not texto or not str(texto).strip():
            raise ErrorHerramienta("Falta la estación. Códigos disponibles: " + ", ".join(self.codigos))
        q = _norm(texto)
        q_cod = q.replace(" ", "-").replace("_", "-")
        if q_cod in self.codigos:
            return q_cod
        sin_guion = {c.replace("-", ""): c for c in self.codigos}
        if q.replace(" ", "").replace("-", "") in sin_guion:
            return sin_guion[q.replace(" ", "").replace("-", "")]
        candidatos = set()
        for pref, muni in MUNICIPIOS.items():
            if _norm(muni) in q:
                candidatos |= {c for c in self.codigos if c.startswith(pref + "-")}
        tokens = [t for t in q.replace("-", " ").split() if len(t) >= 3]
        for c in self.codigos:
            sitio = c.split("-", 1)[1]
            if any(t.startswith(sitio) or sitio.startswith(t) for t in tokens):
                candidatos.add(c)
        if len(candidatos) == 1:
            return candidatos.pop()
        if candidatos:
            raise ErrorHerramienta(f"'{texto}' coincide con varias estaciones: "
                                   f"{', '.join(sorted(candidatos))}. Indica el código exacto.")
        raise ErrorHerramienta(f"No reconozco la estación '{texto}'. Códigos disponibles: "
                               + ", ".join(self.codigos))

    def _fecha(self, fecha: str | None) -> pd.Timestamp | None:
        if fecha is None or str(fecha).strip().lower() in PALABRAS_HOY:
            return None
        try:
            return pd.Timestamp(str(fecha).strip()).normalize()
        except (ValueError, TypeError):
            raise ErrorHerramienta(f"Fecha inválida: '{fecha}'. Usa el formato AAAA-MM-DD.")

    # --- get_pm25 ----------------------------------------------------------------
    def get_pm25(self, estacion: str, fecha: str | None = None) -> dict:
        cod = self.resolver_estacion(estacion)
        serie = self.diario[self.diario["estacion"] == cod].set_index("fecha").sort_index()
        f = self._fecha(fecha)
        nota = None
        if f is None:
            validos = serie[serie["valido"]]
            f = validos.index.max()
            nota = (f"No hay datos en tiempo real: se muestra el dato válido más reciente de {cod} "
                    f"({f.date()}). Los datos descargados del SIATA llegan hasta {self.fecha_max.date()}; "
                    f"hoy es {date.today().isoformat()}.")
        if f not in serie.index:
            raise ErrorHerramienta(
                f"{cod} no tiene datos el {f.date()}. Tiene datos entre {serie.index.min().date()} "
                f"y {serie.index.max().date()}.")
        fila = serie.loc[f]
        valor = float(fila["pm25"])
        cat = ica.categoria(valor)
        out = {
            "estacion": cod, "municipio": MUNICIPIOS.get(cod.split("-")[0]),
            "fecha": f.date().isoformat(),
            "pm25_promedio_diario_ug_m3": round(valor, 1),
            "horas_con_dato": int(fila["n_horas"]), "dia_valido": bool(fila["valido"]),
            "categoria_ica": cat, "color_ica": ica.color(cat),
            "nivel_episodio_res_2254": ica.nivel_episodio(valor),
            "limite_diario_ug_m3": cfg.LIMITE_PM25_24H,
            "supera_limite_diario": valor > cfg.LIMITE_PM25_24H,
            "fuente": FUENTE_DATOS,
            "norma": "Res. 2254 de 2017 (límite diario: Art. 2; ICA: Tabla 6; niveles: Tabla 4)",
        }
        vecinas = self.vecinas_activas(cod)
        if vecinas:
            out["otras_estaciones_activas_del_municipio"] = vecinas
        if not fila["valido"]:
            out["advertencia"] = (f"Solo {int(fila['n_horas'])} horas con dato (se exigen 18 de 24): "
                                  "el promedio es poco confiable.")
        if nota:
            out["nota"] = nota
        return out

    # --- predict_tomorrow --------------------------------------------------------
    def _filas_modelo(self) -> pd.DataFrame:
        if self._dataset is None:
            self._dataset = F.construir_dataset(self.diario)
        return self._dataset

    def predict_tomorrow(self, estacion: str, fecha: str | None = None) -> dict:
        if self.modelo is None:
            raise ErrorHerramienta("No hay modelo entrenado. Corre `python -m src.train`.")
        cod = self.resolver_estacion(estacion)
        ds = self._filas_modelo()
        filas = ds[ds["estacion"] == cod]
        if filas.empty:
            raise ErrorHerramienta(f"{cod} no tiene días con datos suficientes para predecir.")
        f = self._fecha(fecha)
        if f is None:
            fila = filas.loc[filas["fecha_base"].idxmax()]
        else:
            sel = filas[filas["fecha_base"] == f]
            if sel.empty:
                raise ErrorHerramienta(
                    f"No se puede predecir el día siguiente a {f.date()} en {cod}: ese día no tiene "
                    "dato válido o faltan días previos. Prueba otra fecha.")
            fila = sel.iloc[0]
        proba = self.modelo.predict_proba(fila[F.COLUMNAS_X].to_numpy(float)[None, :])[0]
        k = int(proba.argmax())
        out = {
            "estacion": cod, "municipio": MUNICIPIOS.get(cod.split("-")[0]),
            "datos_hasta": fila["fecha_base"].date().isoformat(),
            "fecha_predicha": fila["fecha_objetivo"].date().isoformat(),
            "clase_predicha": F.CLASES[k],
            "probabilidades": {F.CLASES[i]: round(float(p), 3) for i, p in enumerate(proba)},
            "pm25_medido_ultimo_dia_ug_m3": round(float(fila["pm25_hoy"]), 1),
            "clase_ultimo_dia": F.CLASES[int(fila["clase_hoy"])],
            "modelo": "Regresión logística multinomial propia (src/model.py), "
                      f"entrenada con {self.meta_modelo.get('entrenado_con', '?')}",
            "clases": "Buena <= 12; Aceptable 13-37; Dañina para grupos sensibles o peor > 37 µg/m³",
            "que_predice": "Solo la clase del día siguiente; el modelo no estima la concentración.",
        }
        vecinas = self.vecinas_activas(cod)
        if vecinas:
            out["otras_estaciones_activas_del_municipio"] = vecinas
        if self.resumen_modelo:
            out["desempeno_en_prueba"] = self.resumen_modelo
            out["advertencia"] = ("En prueba (2024-2026) el modelo empata con la persistencia en F1 macro; "
                                  "cuando anuncia un día > 37 µg/m³ acierta aproximadamente "
                                  f"{self.resumen_modelo['precision_alertas']:.0%} de las veces.")
        if pd.isna(fila["pm25_manana"]) and fila["fecha_base"] < self.fecha_max:
            out["nota"] = f"{cod} no tiene datos después de {fila['fecha_base'].date()}."
        elif fila["fecha_base"] == self.fecha_max:
            out["nota"] = (f"Predicción para el día siguiente al último dato descargado "
                           f"({self.fecha_max.date()}), no para mañana en tiempo real.")
        return out

    # --- search_docs ----------------------------------------------------------------
    def search_docs(self, consulta: str) -> dict:
        if self.buscador is None:
            raise ErrorHerramienta("No hay índice de documentos. Corre `python -m src.rag construir`.")
        if not consulta or not str(consulta).strip():
            raise ErrorHerramienta("La consulta está vacía.")
        frags = self.buscador.buscar(str(consulta))
        return {
            "consulta": consulta,
            "fragmentos": [{k: f[k] for k in ("cita", "documento", "ubicacion", "titulo", "texto")}
                           for f in frags],
            "instruccion": ("Responde solo con lo que digan estos fragmentos y cita cada afirmación con su "
                            "campo 'cita'. Si no responden la pregunta, dilo explícitamente."),
        }

    # --- despacho ------------------------------------------------------------------
    ARGUMENTOS = {"get_pm25": {"estacion", "fecha"}, "predict_tomorrow": {"estacion", "fecha"},
                  "search_docs": {"consulta"}}

    def ejecutar(self, nombre: str, argumentos: dict) -> dict:
        funciones = {"get_pm25": self.get_pm25, "predict_tomorrow": self.predict_tomorrow,
                     "search_docs": self.search_docs}
        if nombre not in funciones:
            raise ErrorHerramienta(f"Herramienta desconocida: {nombre}. Disponibles: {list(funciones)}")
        permitidos = self.ARGUMENTOS[nombre]
        extra = set(argumentos) - permitidos
        if extra:
            raise ErrorHerramienta(f"Argumentos no válidos para {nombre}: {sorted(extra)}")
        return funciones[nombre](**argumentos)


ESQUEMAS = [
    {"type": "function", "function": {
        "name": "get_pm25",
        "description": ("PM2.5 promedio diario medido en una estación del SIATA en una fecha, con su "
                        "categoría ICA, nivel de episodio y comparación con el límite diario de 37 µg/m³. "
                        "Sin fecha (o 'hoy') devuelve el dato más reciente disponible."),
        "parameters": {"type": "object", "properties": {
            "estacion": {"type": "string", "description": "Código de estación (p. ej. MED-ARAN) o municipio"},
            "fecha": {"type": "string", "description": "AAAA-MM-DD; omitir para el dato más reciente"},
        }, "required": ["estacion"]},
    }},
    {"type": "function", "function": {
        "name": "predict_tomorrow",
        "description": ("Predice con el modelo propio la clase de PM2.5 del día siguiente en una estación "
                        "(Buena, Aceptable o Dañina para grupos sensibles o peor), con probabilidades. "
                        "Sin fecha usa el último día con datos; con fecha predice el día siguiente a esa fecha."),
        "parameters": {"type": "object", "properties": {
            "estacion": {"type": "string", "description": "Código de estación (p. ej. MED-ARAN) o municipio"},
            "fecha": {"type": "string", "description": "AAAA-MM-DD del último día conocido; omitir para el más reciente"},
        }, "required": ["estacion"]},
    }},
    {"type": "function", "function": {
        "name": "search_docs",
        "description": ("Busca en la normativa y documentos oficiales (Res. 2254 de 2017 con sus tablas de ICA y "
                        "niveles, POECA, PIGECA e informe del SIATA). Devuelve fragmentos con su cita "
                        "[documento, ubicación]. Úsala para preguntas sobre normas, niveles de prevención/alerta/"
                        "emergencia, medidas, protocolos, ICA o efectos en la salud."),
        "parameters": {"type": "object", "properties": {
            "consulta": {"type": "string", "description": "Pregunta o términos de búsqueda en español"},
        }, "required": ["consulta"]},
    }},
]
