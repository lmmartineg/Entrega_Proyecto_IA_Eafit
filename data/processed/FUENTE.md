# Fuente de los datos procesados

Los archivos de esta carpeta son **productos derivados**: los generó este proyecto con `python -m src.data process` a partir de los datos originales del SIATA. No son datos oficiales.

**Datos originales:** SIATA, 2026, "Material Particulado - PM2.5", https://doi.org/10.83041/AUWZWT, Repositorio de Datos SIATA (https://datos.siata.gov.co), V5. Titular de los datos: Área Metropolitana del Valle de Aburrá (AMVA). Descargados el 6 nov 2026 (archivos mensuales de 2019-01 a 2026-09).

**Procesamiento propio:**

- Paso del formato ancho (una columna por estación) a formato largo, y descarte de horas sin dato o con valores negativos.
- `pm25_diario.csv`: promedio del día calendario por estación. `valido` indica si el día tiene al menos 18 de 24 horas con dato (regla del 75 % de los informes del SIATA).
- `reporte_unidades.csv`: mediana y percentil 99 por estación y año.
- `estaciones.csv`: primera y última fecha con dato por estación.

Los valores horarios originales no se modifican; solo se agregan. La serie horaria limpia (`pm25_horario.csv`, unos 50 MB) no se sube al repositorio: se regenera con el comando de arriba.

Uso según los Términos y Condiciones del Repositorio de Datos SIATA (V1, borrador, abril 2026): uso académico, sin fines comerciales, citando la fuente.
