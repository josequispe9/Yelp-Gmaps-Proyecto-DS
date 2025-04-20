""" 
ETL para el conjunto de datos de Tráfico Automovilístico de NYC.
"""

# librerias

import pandas as pd
import numpy as np
import geopandas as gpd
import logging
from pathlib import Path
import sys
from io import StringIO
import unicodedata
import re

# --- Configuración de Logging  ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s.%(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger(__name__)

# --- Definición de Rutas y Constantes ---

try:
    RUTA_BASE = Path(r"H:\git\proyecto grupal 2\Yelp-Gmaps-Proyecto-DS").resolve()
    RUTA_RAW = RUTA_BASE / "data" / "raw"
    RUTA_PROCESSED = RUTA_BASE / "data" / "processed"
    RUTA_DIM = RUTA_PROCESSED / "dim"

    RUTA_CSV_TRAFICO = (RUTA_RAW / "trafico automovilistico NY" / "Automated_Traffic_Volume_Counts_20250416.csv").resolve()

    # --- Rutas de SALIDA ---
    RUTA_SALIDA_DIM_SEGMENTO_GPKG = (RUTA_DIM / "dim_segmento_vial.gpkg").resolve()
    RUTA_SALIDA_DIM_SEGMENTO_CSV = (RUTA_DIM / "dim_segmento_vial.csv").resolve()

    RUTA_SALIDA_HECHOS_TRAFICO_CSV = (RUTA_PROCESSED / "fact_volumen_trafico.csv").resolve()
    
    # Creamos directorios 
    RUTA_PROCESSED.mkdir(parents=True, exist_ok=True)
    RUTA_DIM.mkdir(parents=True, exist_ok=True)

    # CRS Esperado 
    CRS_PROYECTADO_NYC = "EPSG:2263"

except Exception as e:
    log.exception(f"Error crítico al definir rutas: {e}")
    sys.exit(1)
# Creamos diccionarios nombres de las columnas traducidas
TRADUCCIONES_COLUMNAS_TRAFICO = {
    'requestid': 'id_solicitud', 'boro': 'distrito', 'yr': 'anio', 'm': 'mes',
    'd': 'dia', 'hh': 'hora', 'mm': 'minuto', 'vol': 'volumen',
    'segmentid': 'id_segmento', 'wktgeom': 'geometria_wkt', 'street': 'calle',
    'fromst': 'calle_desde', 'tost': 'calle_hasta', 'direction': 'direccion_flujo'
}

# --- Funciones Auxiliares ---

"""Estandarizamos el nombre de las columnas a formato snake_case."""
def limpiar_nombre_columna(col_name: str) -> str:

    # ... (igual que antes) ...
    if not isinstance(col_name, str): col_name = str(col_name)
    try:
        col_name = col_name.replace('(', ' ').replace(')', ' ').replace(':', '_')
        normalized = unicodedata.normalize('NFKD', col_name).encode('ASCII', 'ignore').decode('utf-8').lower()
        snake_case = re.sub(r'\s+', '_', normalized)
        snake_case = re.sub(r'[^a-z0-9_]+', '', snake_case)
        snake_case = snake_case.strip('_')
        if not snake_case: log.warning(f"'{col_name}' vacío tras limpieza."); return f"col_limpia_{hash(col_name)}"
        return snake_case
    except Exception as e: log.warning(f"No se pudo limpiar '{col_name}': {e}."); return col_name

# --- Funciones de Procesamiento para Tráfico ---
"""Cargamos los datos del CSV de Tráfico (asume header=0)."""
def cargar_datos_trafico(ruta: Path) -> pd.DataFrame | None:
    
    # ... (igual que antes) ...
    log.info(f"Intentando cargar datos Tráfico desde: {ruta}")
    if not ruta.is_file(): log.error(f"Ruta Tráfico no es archivo: {ruta}"); return None
    df = None
    try:
        dtypes_inicial = {'Yr': str, 'M': str, 'D': str, 'HH': str, 'MM': str}
        df = pd.read_csv(ruta, encoding='utf-8', header=0, dtype=dtypes_inicial, low_memory=False)
        log.info(f"Archivo Tráfico '{ruta.name}' cargado con UTF-8 (header=0).")
    except UnicodeDecodeError:
        log.warning(f"Fallo UTF-8 Tráfico '{ruta.name}'. Intentando Latin-1...")
        try:
            df = pd.read_csv(ruta, encoding='latin1', header=0, dtype=dtypes_inicial, low_memory=False)
            log.warning(f"Archivo Tráfico '{ruta.name}' cargado con Latin-1 (header=0).")
        except Exception as e_latin: log.exception(f"Fallo Tráfico Latin-1: {e_latin}"); return None
    except Exception as e_main: log.exception(f"Error carga Tráfico: {e_main}"); return None
    if df is None or df.empty: log.warning("DataFrame Tráfico vacío/None tras carga."); return df
    log.info(f"Carga Tráfico: {df.shape[0]} filas, {df.shape[1]} cols.")
    log.debug(f"Columnas originales Tráfico: {df.columns.tolist()}")
    return df

"""Pipeline de transformación Tráfico, incluimos conversión a GeoDataFrame."""
def transformar_trafico_y_geometria(df: pd.DataFrame) -> gpd.GeoDataFrame | pd.DataFrame:

    # ... (igual que antes) ...
    if df is None or df.empty: log.error("Input inválido."); return pd.DataFrame()
    log.info("Inicio transformación Tráfico (incluye geoespacial)...")
    try: df.columns = [limpiar_nombre_columna(col) for col in df.columns]
    except Exception as e: log.exception("Error limpieza nombres Tráfico."); return pd.DataFrame()
    log.info(f"Columnas Tráfico limpiadas: {df.columns.tolist()}")
    try:
        nuevos_nombres = [TRADUCCIONES_COLUMNAS_TRAFICO.get(col, col) for col in df.columns]
        renombradas_count = sum(1 for old, new in zip(df.columns, nuevos_nombres) if old != new)
        df.columns = nuevos_nombres
        if renombradas_count == 0: log.warning("Ninguna col Tráfico renombrada.")
        else: log.info(f"{renombradas_count} cols Tráfico renombradas.")
        log.info(f"Nombres finales Tráfico: {df.columns.tolist()}")
    except Exception as e: log.exception("Error renombrado manual Tráfico."); return pd.DataFrame()
    log.info("Creando columna 'fecha_hora'...")
    cols_fecha = ['anio', 'mes', 'dia', 'hora', 'minuto']
    if not all(col in df.columns for col in cols_fecha): log.error(f"Faltan cols fecha."); return pd.DataFrame()
    try:
        for col in cols_fecha: df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=cols_fecha, inplace=True)
        for col in cols_fecha: df[col] = df[col].astype(int)
        datetime_dict = {'year': df['anio'], 'month': df['mes'], 'day': df['dia'], 'hour': df['hora'], 'minute': df['minuto']}
        df['fecha_hora'] = pd.to_datetime(datetime_dict, errors='coerce')
        original_rows = len(df); df.dropna(subset=['fecha_hora'], inplace=True)
        if len(df) < original_rows: log.warning(f"Eliminadas {original_rows - len(df)} filas con fechas inválidas.")
        df.drop(columns=cols_fecha, inplace=True); log.info("'fecha_hora' creada.")
    except Exception as e: log.exception("Error creando 'fecha_hora'."); return pd.DataFrame()
    log.info("Aplicando otras conversiones de tipo y limpieza...")
    try:
        if 'id_solicitud' in df.columns: df['id_solicitud'] = df['id_solicitud'].astype('string')
        if 'id_segmento' in df.columns: df['id_segmento'] = df['id_segmento'].astype('string')
        if 'volumen' in df.columns: df['volumen'] = pd.to_numeric(df['volumen'], errors='coerce').astype('Int64')
        if 'distrito' in df.columns: df['distrito'] = df['distrito'].astype(str).str.title().astype('category')
        if 'direccion_flujo' in df.columns: df['direccion_flujo'] = df['direccion_flujo'].astype(str).str.upper().str.strip().astype('category')
        for col in ['calle', 'calle_desde', 'calle_hasta']:
            if col in df.columns: df[col] = df[col].astype(str).str.strip().astype('string')
    except Exception as e: log.warning(f"Error conversiones tipo adicionales: {e}")
    log.info("Procesando geometría WKT...")
    col_wkt = 'geometria_wkt'; gdf = df
    if col_wkt in df.columns:
        try:
            log.info(f"Convirtiendo WKT a geometría con CRS: {CRS_PROYECTADO_NYC}")
            geometry_col = gpd.GeoSeries.from_wkt(df[col_wkt], crs=CRS_PROYECTADO_NYC, on_invalid='ignore')
            gdf = gpd.GeoDataFrame(df, geometry=geometry_col, crs=CRS_PROYECTADO_NYC)
            gdf = gdf.drop(columns=[col_wkt]); log.info("Conversión a GeoDataFrame OK.")
            valid_geom = gdf.geometry.is_valid.sum()
            if valid_geom < len(gdf): log.warning(f"{len(gdf) - valid_geom} geometrías inválidas/ignoradas.")
        except ImportError: log.error("¡GeoPandas no encontrado! No se procesó geometría."); gdf = df
        except Exception as e: log.exception("Error convirtiendo WKT."); gdf = df
    else: log.warning(f"Columna WKT '{col_wkt}' no encontrada.");
    log.info("Transformación Tráfico completada.")
    return gdf


#Funcion para guardar los dattos resulttantes en un archivo CSV y un GeoPackage.
def crear_y_guardar_dimension_segmento(
    gdf_trafico: gpd.GeoDataFrame | pd.DataFrame,
    ruta_salida_gpkg: Path,
    ruta_salida_csv: Path # <-- Nuevo argumento para la ruta CSV
):
    log.info("Inicio creación dimensión segmentos viales...")
    cols_segmento_base = ['id_segmento', 'calle', 'calle_desde', 'calle_hasta', 'direccion_flujo', 'distrito']
    cols_segmento = cols_segmento_base.copy() # Copiar para modificar
    is_geodataframe = False # Asumir que no es geo inicialmente

    # Comprobamos si tenemos geometría válida
    if 'geometry' in gdf_trafico.columns and isinstance(gdf_trafico, gpd.GeoDataFrame):
        # Verificamos si la columna de geometría NO está vacía o toda nula
        if not gdf_trafico['geometry'].isnull().all():
            cols_segmento.append('geometry')
            is_geodataframe = True
            log.info("Incluyendo columna 'geometry' en la dimensión segmento.")
        else:
             log.warning("Columna 'geometry' existe pero está vacía/nula. Dimensión se creará sin geometría.")
    else:
        log.warning("Columna 'geometry' no encontrada o no es GeoDataFrame. Dimensión se creará sin geometría.")

    # Verificamos columnas base
    if not all(col in gdf_trafico.columns for col in cols_segmento_base):
        log.error(f"Faltan columnas base para dim_segmento. Requeridas: {cols_segmento_base}")
        return

    try:
        # otorgamosi ids únicos a los segmentos
        dim_segmento = gdf_trafico[cols_segmento].copy()
        log.info(f"Registros de segmento antes de duplicados: {len(dim_segmento)}")
        if not dim_segmento.dropna(subset=['id_segmento'])['id_segmento'].is_unique:
             log.warning(f"'id_segmento' no es único. Se mantendrá la primera aparición por ID.")
        dim_segmento.drop_duplicates(subset=['id_segmento'], keep='first', inplace=True, ignore_index=True)
        log.info(f"Dimensión segmento creada con {len(dim_segmento)} segmentos únicos.")

        # Aseguramos tipos finales 
        dim_segmento['id_segmento'] = dim_segmento['id_segmento'].astype('string')
        if 'distrito' in dim_segmento.columns: dim_segmento['distrito'] = dim_segmento['distrito'].astype('category')
        if 'direccion_flujo' in dim_segmento.columns: dim_segmento['direccion_flujo'] = dim_segmento['direccion_flujo'].astype('category')
        # ... otros tipos si es necesario ...

        # Reordenamos las columnas para salida
        cols_ordenadas = ['id_segmento', 'calle', 'calle_desde', 'calle_hasta', 'direccion_flujo', 'distrito', 'geometry']
        cols_finales_dim = [col for col in cols_ordenadas if col in dim_segmento.columns]
        cols_otras_dim = [col for col in dim_segmento.columns if col not in cols_finales_dim]
        dim_segmento_guardar = dim_segmento[cols_finales_dim + cols_otras_dim]

        # Guardar en CSV y GeoPackage 
        guardado_exitoso = False
        if not dim_segmento_guardar.empty:
            # Guardar GeoPackage (si es GeoDataFrame)
            if is_geodataframe:
                try:
                    log.info(f"Guardando dimensión segmento como GeoPackage en: {ruta_salida_gpkg}")
                    dim_segmento_guardar.to_file(ruta_salida_gpkg, driver='GPKG', layer='segmentos_viales')
                    log.info("Dimensión segmento GeoPackage guardada.")
                    guardado_exitoso = True # Marcamos éxito si al menos GPKG se guardó
                except ImportError: log.error("¡Falta Fiona/GeoPandas! No se pudo guardar GeoPackage.")
                except Exception as e_gpkg: log.exception(f"Error guardando dim_segmento GPKG: {e_gpkg}")

            # Guardar CSV 
            try:
                log.info(f"Guardando dimensión segmento como CSV en: {ruta_salida_csv}")
                # La columna 'geometry' se convertirá a WKT automáticamente si existe
                dim_segmento_guardar.to_csv(ruta_salida_csv, index=False, encoding='utf-8-sig')
                log.info("Dimensión segmento CSV guardada.")
                guardado_exitoso = True
            except Exception as e_csv:
                log.exception(f"Error guardando dim_segmento CSV: {e_csv}")
                # Si GPKG falla Y CSV también falla, el éxito general será False

        else:
            log.warning("Dimensión segmento vacía, no se guardaron archivos.")

    except Exception as e:
        log.exception(f"Error inesperado creando la dimensión segmento: {e}")

"""Creamos  y guardamos la tabla de hechos de volumen de tráfico como CSV."""
def crear_y_guardar_hechos_trafico(df_trafico: pd.DataFrame | gpd.GeoDataFrame, ruta_salida: Path):

    log.info("Inicio creación tabla de hechos de tráfico...")
    cols_hechos = ['id_solicitud', 'fecha_hora', 'volumen', 'id_segmento']
    if not all(col in df_trafico.columns for col in cols_hechos): log.error(f"Faltan columnas para tabla de hechos."); return
    try:
        df_hechos = df_trafico[cols_hechos].copy()
        df_hechos['id_solicitud'] = df_hechos['id_solicitud'].astype('string')
        df_hechos['id_segmento'] = df_hechos['id_segmento'].astype('string')
        # El volumen ya debería ser Int64, fecha_hora ya datetime64[ns]
        log.info(f"Tabla de hechos creada: {len(df_hechos)} registros.")
        if not df_hechos.empty:
            log.info(f"Guardando tabla de hechos CSV en: {ruta_salida}")
            df_hechos.to_csv(ruta_salida, index=False, encoding='utf-8-sig')
            log.info("Tabla de hechos CSV guardada.")
        else: log.warning("Tabla de hechos vacía, no se guardó.")
    except Exception as e: log.exception(f"Error creando/guardando tabla de hechos: {e}")

# --- Flujo Principal ---

"""Orquestamos el flujo ETL Tráfico: Dimensión Segmento (GPKG+CSV) y Hechos (CSV)."""
def main_trafico():
    
    log.info("================================================")
    log.info("=== INICIO ETL: Tráfico Vehicular NYC (Salida Doble Dim) ===")
    log.info("================================================")

    # 0. Validación Entrada
    log.info(f"Verificando entrada Tráfico: {RUTA_CSV_TRAFICO}")
    if not RUTA_CSV_TRAFICO.is_file(): log.critical(f"Abortando: Archivo Tráfico no encontrado: {RUTA_CSV_TRAFICO}"); sys.exit(1)
    log.info("Entrada Tráfico encontrada.")

    # 1. Extracción
    log.info("--- Fase 1: Extracción Tráfico ---")
    df_raw_trafico = cargar_datos_trafico(RUTA_CSV_TRAFICO)
    if df_raw_trafico is None or df_raw_trafico.empty: log.critical("Fallo extracción Tráfico."); sys.exit(1)

    # 2. Transformación 
    log.info("--- Fase 2: Transformación Tráfico (incluye Geoespacial) ---")
    gdf_processed_trafico = transformar_trafico_y_geometria(df_raw_trafico.copy())
    if gdf_processed_trafico.empty: log.critical("Transformación Tráfico vacía."); sys.exit(1)
    log.info(f"Tipo de dato tras transformación: {type(gdf_processed_trafico)}")

    # 3. Creamos y guardamos Dimensión Segmento Vial
    log.info("--- Fase 3: Creación/Guardado Dimensión Segmento Vial (GPKG y CSV) ---")
    # Llamamos a la función modificada pasándole AMBAS rutas de salida
    crear_y_guardar_dimension_segmento(
        gdf_processed_trafico,
        RUTA_SALIDA_DIM_SEGMENTO_GPKG,
        RUTA_SALIDA_DIM_SEGMENTO_CSV
    )

    # 4. Creamos y guardamos las tablas de echos
    log.info("--- Fase 4: Creación/Guardado Tabla de Hechos Volumen ---")
    crear_y_guardar_hechos_trafico(gdf_processed_trafico, RUTA_SALIDA_HECHOS_TRAFICO_CSV)

    log.info("===================================================")
    log.info("=== ETL Tráfico Vehicular completado ===")
    log.info(f"Dimensión Segmento -> {RUTA_SALIDA_DIM_SEGMENTO_GPKG.name} Y {RUTA_SALIDA_DIM_SEGMENTO_CSV.name}")
    log.info(f"Tabla de Hechos -> {RUTA_SALIDA_HECHOS_TRAFICO_CSV.name}")
    log.info("===================================================")

if __name__ == "__main__":
    main_trafico()