

# Librerias
import pandas as pd
import numpy as np
import geopandas as gpd
import re
import unicodedata
import logging
from pathlib import Path
import sys
from io import StringIO

# --- Configuración de Logging  ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s.%(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger(__name__)

# --- Definición de Rutas y Constantes ---
try:
    RUTA_BASE = Path(r"H:\git\proyecto grupal 2\Yelp-Gmaps-Proyecto-DS")
    RUTA_RAW = RUTA_BASE / "data" / "raw"
    RUTA_PROCESSED = RUTA_BASE / "data" / "processed"
    RUTA_DIM = RUTA_PROCESSED / "dim"

    # Rutas NTA
    RUTA_CSV_NTA = (RUTA_RAW / "Áreas de Tabulación de Vecindarios" / "2020_Neighborhood_Tabulation_Areas__NTAs__20250416.csv").resolve()
    # *** Definimos AMBAS rutas de salida para NTA ***
    RUTA_SALIDA_NTA_GPKG = (RUTA_PROCESSED / "nta_2020_processed.gpkg").resolve() # Salida GeoPackage
    RUTA_SALIDA_NTA_CSV = (RUTA_PROCESSED / "nta_2020_processed.csv").resolve()  # Salida CSV

    # Ruta para Dimensión Distrito (sigue siendo CSV)
    RUTA_SALIDA_DIM_DISTRITO = (RUTA_DIM / "dim_distrito.csv").resolve()

    # Creamos los directorios de salida necesarios
    RUTA_PROCESSED.mkdir(parents=True, exist_ok=True)
    RUTA_DIM.mkdir(parents=True, exist_ok=True)

except Exception as e:
    log.exception(f"Error crítico al definir o resolver rutas: {e}")
    sys.exit(1)

# --- Diccionario de Traducción para NTA ---
TRADUCCIONES_COLUMNAS_NTA = {
    'the_geom': 'geometria_wkt', 'borocode': 'codigo_distrito', 'boroname': 'nombre_distrito',
    'countyfips': 'fips_condado', 'nta2020': 'id_nta_2020', 'ntaname': 'nombre_nta',
    'ntaabbrev': 'abreviatura_nta', 'ntatype': 'tipo_nta', 'cdta2020': 'id_cdta_2020',
    'cdtaname': 'nombre_cdta', 'shape_leng': 'longitud_perimetro', 'shape_area': 'area_poligono'
}

# --- Funciones Auxiliares --- 
"""Estandariza un nombre de columna a formato snake_case."""
def limpiar_nombre_columna(col_name: str) -> str:
    if not isinstance(col_name, str): col_name = str(col_name)
    try:
        normalized = unicodedata.normalize('NFKD', col_name).encode('ASCII', 'ignore').decode('utf-8').lower()
        snake_case = re.sub(r'\s+', '_', normalized)
        snake_case = re.sub(r'[^a-z0-9_]+', '', snake_case)
        snake_case = snake_case.strip('_')
        if not snake_case: log.warning(f"'{col_name}' vacío tras limpieza."); return f"col_limpia_{hash(col_name)}"
        return snake_case
    except Exception as e: log.warning(f"No se pudo limpiar '{col_name}': {e}."); return col_name

# --- Funciones de Procesamiento NTA ---
def cargar_datos_csv_nta(ruta: Path) -> pd.DataFrame | None:
    """Carga datos del CSV de NTA (header=0)."""
    # ... (código igual que antes) ...
    log.info(f"Intentando cargar datos NTA desde: {ruta}")
    if not ruta.is_file(): log.error(f"Ruta NTA no es archivo: {ruta}"); return None
    df = None
    try:
        df = pd.read_csv(ruta, encoding='utf-8', header=0, low_memory=False)
        log.info(f"Archivo NTA '{ruta.name}' cargado con UTF-8 (header=0).")
    except UnicodeDecodeError:
        log.warning(f"Fallo UTF-8 NTA '{ruta.name}'. Intentando Latin-1...")
        try:
            df = pd.read_csv(ruta, encoding='latin1', header=0, low_memory=False)
            log.warning(f"Archivo NTA '{ruta.name}' cargado con Latin-1 (header=0).")
        except Exception as e_latin: log.exception(f"Fallo NTA Latin-1: {e_latin}"); return None
    except Exception as e_main: log.exception(f"Error carga NTA: {e_main}"); return None
    if df is None or df.empty: log.warning("DataFrame NTA vacío/None tras carga."); return df
    log.info(f"Carga NTA: {df.shape[0]} filas, {df.shape[1]} cols.")
    log.debug(f"Columnas originales NTA: {df.columns.tolist()}")
    return df
"""Pipeline de transformación NTA, incluyendo conversión a GeoDataFrame."""

def transformar_datos_nta(df: pd.DataFrame) -> gpd.GeoDataFrame | pd.DataFrame:
    # ... (código igual que antes, convierte WKT a geometría y devuelve GeoDataFrame) ...
    if df is None or df.empty: log.error("Input inválido."); return pd.DataFrame()
    log.info("Inicio transformación NTA (incluye geoespacial)...")
    try: df.columns = [limpiar_nombre_columna(col) for col in df.columns]
    except Exception as e: log.exception("Error limpieza nombres NTA."); return pd.DataFrame()
    log.info(f"Columnas NTA limpiadas: {df.columns.tolist()}")
    try:
        nuevos_nombres = [TRADUCCIONES_COLUMNAS_NTA.get(col, col) for col in df.columns]
        renombradas_count = sum(1 for old, new in zip(df.columns, nuevos_nombres) if old != new)
        df.columns = nuevos_nombres
        if renombradas_count == 0: log.warning("Ninguna columna NTA renombrada.")
        else: log.info(f"{renombradas_count} cols NTA renombradas. Finales: {df.columns.tolist()}")
    except Exception as e: log.exception("Error renombrado manual NTA."); return pd.DataFrame()
    log.info("Aplicando conversiones de tipo NTA...")
    columnas_a_convertir = {
        'codigo_distrito': 'category', 'nombre_distrito': 'category', 'fips_condado': 'string',
        'id_nta_2020': 'string', 'nombre_nta': 'string', 'abreviatura_nta': 'string',
        'tipo_nta': 'category', 'id_cdta_2020': 'string', 'nombre_cdta': 'string',
        'longitud_perimetro': 'float', 'area_poligono': 'float'
    }
    for col, tipo in columnas_a_convertir.items():
        if col in df.columns:
            try:
                if tipo in ['category', 'string']: df[col] = df[col].astype(str).str.strip().replace(['nan','None','','Missing'], np.nan if tipo=='category' else '')
                if tipo == 'float': df[col] = pd.to_numeric(df[col], errors='coerce')
                elif tipo == 'category': df[col] = df[col].astype('category')
                elif tipo == 'string': df[col] = df[col].astype('string')
                else: df[col] = df[col].astype(tipo)
                log.debug(f"Col NTA '{col}' -> '{tipo}'.")
            except Exception as e: log.warning(f"Error conversión NTA '{col}' a '{tipo}': {e}")
    log.info("Convirtiendo WKT a geometría...")
    col_wkt = 'geometria_wkt'; gdf = df # Empezar asumiendo que devolveremos df normal
    if col_wkt in df.columns:
        try:
            geometry_col = gpd.GeoSeries.from_wkt(df[col_wkt], crs="EPSG:4326", on_invalid='ignore')
            gdf = gpd.GeoDataFrame(df, geometry=geometry_col, crs="EPSG:4326")
            gdf = gdf.drop(columns=[col_wkt])
            log.info(f"Col '{col_wkt}' -> 'geometry'. Conversión a GeoDataFrame OK.")
            valid_geom = gdf.geometry.is_valid.sum(); total_geom = len(gdf)
            if valid_geom < total_geom: log.warning(f"{total_geom - valid_geom} geometrías inválidas/ignoradas.")
        except ImportError: log.error("GeoPandas no encontrado. No se procesó geometría."); gdf = df
        except Exception as e: log.exception("Error conversión WKT."); gdf = df
    else: log.warning(f"Columna WKT '{col_wkt}' no encontrada."); gdf = df
    col_pk = 'id_nta_2020'
    if col_pk in gdf.columns:
        if not gdf[col_pk].dropna().is_unique: log.warning(f"'{col_pk}' no es única (ignorando nulos).")
        else: log.info(f"'{col_pk}' es única (ignorando nulos).")
    log.info("Transformación NTA completada.")
    return gdf

"""Extraemos y guardamos el archivo de dimensión de distritos como CSV."""
def crear_y_guardar_dimension_distrito(df_input: pd.DataFrame | gpd.GeoDataFrame, ruta_salida_dim: Path):
    
    # ... (código igual que antes) ...
    log.info("Inicio creación dimensión distritos...")
    cols_distrito = ['codigo_distrito', 'nombre_distrito']
    if not all(col in df_input.columns for col in cols_distrito):
        log.error(f"Faltan {cols_distrito} para crear dim_distrito."); return
    try:
        dim_distrito = df_input[cols_distrito].copy()
        dim_distrito.drop_duplicates(inplace=True); dim_distrito.dropna(inplace=True)
        dim_distrito.sort_values(by='codigo_distrito', inplace=True); dim_distrito.reset_index(drop=True, inplace=True)
        dim_distrito['codigo_distrito'] = pd.to_numeric(dim_distrito['codigo_distrito'], errors='coerce').astype('Int64')
        dim_distrito['nombre_distrito'] = dim_distrito['nombre_distrito'].astype('string')
        dim_distrito.dropna(subset=['codigo_distrito'], inplace=True)
        log.info(f"Dimensión distrito creada: {len(dim_distrito)} registros.")
        if not dim_distrito.empty:
            try:
                dim_distrito.to_csv(ruta_salida_dim, index=False, encoding='utf-8-sig')
                log.info(f"Dimensión distrito guardada en: {ruta_salida_dim}")
            except Exception as e_save: log.exception(f"Error guardando dim_distrito: {e_save}")
        else: log.warning("Dimensión distrito vacía, no se guardó.")
    except Exception as e_create: log.exception(f"Error creando dim_distrito: {e_create}")

"""
Guardamos el (Geo)DataFrame NTA procesado en formato GeoPackage Y CSV.

 """

# --- Función de Guardado Modificada para Múltiples Formatos ---
def guardar_salidas_nta(gdf: gpd.GeoDataFrame | pd.DataFrame, ruta_salida_gpkg: Path, ruta_salida_csv: Path):
 
    log.info("Inicio del guardado de salidas NTA (GPKG y CSV)...")

    if gdf is None or gdf.empty:
        log.warning("(Geo)DataFrame NTA final vacío. No se guardarán archivos.")
        return

    # --- Reordenamos las columnas (Común para ambos formatos) ---
  
    columnas_ordenadas = [
        'id_nta_2020', 'nombre_nta', 'abreviatura_nta', 'tipo_nta',
        'codigo_distrito', 'nombre_distrito', 'fips_condado',
        'id_cdta_2020', 'nombre_cdta',
        'longitud_perimetro', 'area_poligono',
        'geometry' 
    ]
    # Creamos la lista final de columnas en el orden preferido + las restantes
    cols_finales = [col for col in columnas_ordenadas if col in gdf.columns]
    cols_otras = [col for col in gdf.columns if col not in cols_finales]
    gdf_guardar = gdf[cols_finales + cols_otras]

    # --- Guardamos como GeoPackage ---
    if isinstance(gdf_guardar, gpd.GeoDataFrame): # Solo guardar GPKG si es GeoDataFrame
        try:
            log.info(f"Intentando guardar GeoPackage en: {ruta_salida_gpkg}")
            gdf_guardar.to_file(ruta_salida_gpkg, driver='GPKG', layer='NTA_2020')
            log.info(f"Datos NTA procesados guardados como GeoPackage.")
        except ImportError:
             log.error("¡Falta Fiona/GeoPandas! No se pudo guardar a GeoPackage.")
        except Exception as e_gpkg:
             log.exception(f"Error CRÍTICO guardando NTA GeoPackage: {e_gpkg}")
    else:
        log.warning("La entrada no es un GeoDataFrame, omitiendo guardado de GeoPackage.")

    # --- Guardamos como CSV ---
    # Para CSV, la columna 'geometry' (si existe) se convertirá a WKT automáticamente.
    try:
        log.info(f"Intentando guardar CSV en: {ruta_salida_csv}")
        gdf_guardar.to_csv(ruta_salida_csv, index=False, encoding='utf-8-sig')
        log.info(f"Datos NTA procesados guardados también como CSV.")
    except Exception as e_csv:
        log.exception(f"Error CRÍTICO guardando NTA CSV: {e_csv}")


# --- Flujo Principal de Ejecución  ---
def main_nta():
    """Orquesta el flujo ETL NTA, geometría, dim distrito y guarda en GPKG y CSV."""
    log.info("================================================")
    log.info("=== INICIO ETL: NTA 2020 (Geoespacial) y Dim Distrito ===")
    log.info("================================================")

    # 0. Validación Entrada
    log.info(f"Verificando entrada NTA: {RUTA_CSV_NTA}")
    if not RUTA_CSV_NTA.is_file(): log.critical(f"Abortando: Archivo NTA no encontrado: {RUTA_CSV_NTA}"); sys.exit(1)
    log.info("Entrada NTA encontrada.")

    # 1. Extracción
    log.info("--- Fase 1 NTA: Extracción ---")
    df_raw_nta = cargar_datos_csv_nta(RUTA_CSV_NTA)
    if df_raw_nta is None or df_raw_nta.empty: log.critical("Fallo extracción NTA. Abortando."); sys.exit(1)

    # 2. Transformación (resulta en GeoDataFrame o DataFrame)
    log.info("--- Fase 2 NTA: Transformación (incluye Geoespacial) ---")
    gdf_processed_nta = transformar_datos_nta(df_raw_nta.copy())
    if gdf_processed_nta.empty: log.critical("Transformación NTA vacía. Abortando."); sys.exit(1)
    log.info(f"Tipo de dato tras transformación: {type(gdf_processed_nta)}")

    # 3. Carga NTA (Guardado en AMBOS formatos)
    log.info("--- Fase 3 NTA: Carga (Guardado GPKG y CSV) ---")
    # Llamar a la nueva función de guardado múltiple
    guardar_salidas_nta(gdf_processed_nta, RUTA_SALIDA_NTA_GPKG, RUTA_SALIDA_NTA_CSV)

    # 4. Crear y Guardar Dimensión Distrito
    log.info("--- Fase 4: Creación Dimensión Distrito ---")
    crear_y_guardar_dimension_distrito(gdf_processed_nta, RUTA_SALIDA_DIM_DISTRITO)

    log.info("===================================================")
    log.info("=== ETL NTA (Geoespacial) y Dimensión Distrito completados ===")
    log.info("===================================================")

    log.info("Info del (Geo)DataFrame NTA procesado final:")
    buffer = StringIO(); gdf_processed_nta.info(buf=buffer); log.info(buffer.getvalue())


if __name__ == "__main__":
    # Punto de entrada
    main_nta()