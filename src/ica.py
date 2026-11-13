"""Índice de Calidad del Aire (ICA) y niveles de episodio para PM2.5.

Fuente: Resolución 2254 de 2017, MADS.
https://normas.cra.gov.co/gestor/docs/resolucion_minambienteds_2254_2017.htm

- ICA_PM25: Tabla 6 (Art. 20, puntos de corte del ICA), columna "PM2.5 µg/m³
  24 horas"; nombres y colores de la Tabla 5 (Art. 19).
- NIVELES_PM25: Tabla 4 (Art. 10), niveles de prevención, alerta o emergencia
  para PM2.5 en 24 h, aplicables desde el 1 jul 2018 (nota ** de la tabla).

En la página de la CRA las tablas son imágenes (graficas/..._obj_5.gif = Tabla 4,
obj_6.gif = Tabla 5, obj_7.gif = Tabla 6). Se transcribieron de esas imágenes
el 6 nov 2026 y se cotejaron el 12 nov 2026 contra la compilación de la Alcaldía
de Bogotá (SISJUR, Norma1.jsp?i=82634): coinciden. No se cotejó el PDF del Diario Oficial.

Bordes: la tabla usa enteros (0-12, 13-37, ...). Un valor en el hueco, como
12,4, se asigna al rango superior. Así el límite superior de cada rango es
inclusivo y "Aceptable" termina exactamente en 37 µg/m³, el máximo diario
permisible (Art. 2). Es decir, superar la norma implica salir de "Aceptable".

Simplificación del proyecto: la norma calcula el ICA y los niveles con medias
móviles de 24 h (Art. 11 y Art. 21). Aquí se aplican al promedio del día
calendario (00:00-23:00), que es la unidad que predice el modelo.
"""
from __future__ import annotations

# (límite inferior, límite superior) en µg/m³, categoría (Tabla 6) y color (Tabla 5).
# Rangos del índice: 0-50, 51-100, 101-150, 151-200, 201-300, 301-500.
ICA_PM25: list[tuple[float, float, str, str]] = [
    (0, 12, "Buena", "Verde"),
    (13, 37, "Aceptable", "Amarillo"),
    (38, 55, "Dañina a la salud de grupos sensibles", "Naranja"),
    (56, 150, "Dañina a la salud", "Rojo"),
    (151, 250, "Muy dañina a la salud", "Púrpura"),
    (251, 500, "Peligrosa", "Marrón"),
]

# Códigos de color para dibujar. La Res. 2254 solo nombra los colores (Tabla 5); el ICA
# usa los mismos rangos que el AQI de la US EPA (0-50, 51-100, ...), así que se toman
# los valores RGB de la convención de la EPA/AirNow.
COLORES_HEX = {"Verde": "#00E400", "Amarillo": "#FFFF00", "Naranja": "#FF7E00",
               "Rojo": "#FF0000", "Púrpura": "#8F3F97", "Marrón": "#7E0023"}

# Tabla 5 (Art. 19): rango del índice, color, estado y efectos (transcrito de la imagen obj_6.gif).
EFECTOS_ICA: list[tuple[str, str, str, str]] = [
    ("0-50", "Verde", "Buena", "La contaminación atmosférica supone un riesgo bajo para la salud."),
    ("51-100", "Amarillo", "Aceptable", "Posibles síntomas respiratorios en grupos poblacionales sensibles."),
    ("101-150", "Naranja", "Dañina a la salud de grupos sensibles",
     "Los grupos poblacionales sensibles pueden presentar efectos sobre la salud. "
     "1) Ozono troposférico: las personas con enfermedades pulmonares, niños, adultos mayores y las que "
     "constantemente realizan actividad física al aire libre deben reducir su exposición a los "
     "contaminantes del aire. 2) Material particulado: las personas con enfermedad cardiaca o pulmonar, "
     "los adultos mayores y los niños se consideran sensibles y por lo tanto en mayor riesgo."),
    ("151-200", "Rojo", "Dañina para la salud",
     "Todos los individuos pueden comenzar a experimentar efectos sobre la salud. Los grupos sensibles "
     "pueden experimentar efectos más graves para la salud."),
    ("201-300", "Púrpura", "Muy dañina para la salud",
     "Estado de alerta que significa que todos pueden experimentar efectos más graves para la salud."),
    ("301-500", "Marrón", "Peligroso",
     "Advertencia sanitaria. Toda la población puede presentar efectos adversos graves en la salud humana "
     "y están propensos a verse afectados por graves efectos sobre la salud."),
]

# Tabla 4: (límite inferior, límite superior, nivel). Emergencia es ">= 151".
NIVELES_PM25: list[tuple[float, float, str]] = [
    (38, 55, "Prevención"),
    (56, 150, "Alerta"),
    (151, float("inf"), "Emergencia"),
]


def validar_tabla(tabla: list[tuple[float, float, str, str]]) -> None:
    """Falla con un mensaje claro si la tabla está vacía o mal formada."""
    if not tabla:
        raise ValueError(
            "ICA_PM25 está vacía. Transcribe la Tabla 6 de la Res. 2254 de 2017 "
            "en src/ica.py antes de usar categorías ICA."
        )
    for i, (lo, hi, _, _) in enumerate(tabla):
        if lo >= hi:
            raise ValueError(f"Fila {i}: el límite inferior debe ser menor que el superior.")
        if i > 0 and lo < tabla[i - 1][1]:
            raise ValueError(f"Fila {i}: se superpone con la fila anterior.")


def categoria(concentracion: float, tabla: list | None = None) -> str:
    """Devuelve la categoría ICA de una concentración diaria de PM2.5."""
    tabla = ICA_PM25 if tabla is None else tabla
    validar_tabla(tabla)
    for lo, hi, nombre, _ in tabla:
        if lo <= concentracion <= hi:
            return nombre
    # Valores entre dos rangos por redondeo (p. ej. 12.05 si un rango acaba en
    # 12.0 y el siguiente empieza en 12.1) se asignan al rango superior.
    for lo, _, nombre, _ in tabla:
        if concentracion < lo:
            return nombre
    return tabla[-1][2]


def color(nombre_categoria: str, tabla: list | None = None) -> str:
    """Color oficial (Tabla 5) de una categoría ICA."""
    tabla = ICA_PM25 if tabla is None else tabla
    for _, _, nombre, c in tabla:
        if nombre == nombre_categoria:
            return c
    raise KeyError(f"Categoría desconocida: {nombre_categoria}")


def nivel_episodio(concentracion: float, tabla: list | None = None) -> str | None:
    """Nivel de la Tabla 4 (Prevención, Alerta, Emergencia) o None si es menor a 38.
    Los huecos entre rangos (p. ej. 55,5) se asignan al nivel superior, como en el ICA."""
    tabla = NIVELES_PM25 if tabla is None else tabla
    # La tabla usa enteros: el rango anterior a Prevención termina en 38 - 1 = 37.
    if concentracion <= tabla[0][0] - 1:
        return None
    for _, hi, nombre in tabla:
        if concentracion <= hi:
            return nombre
    return tabla[-1][2]
