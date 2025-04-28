"""
ETL para Entradas y Salidas de Metro de NYC
- Carga datos de Entradas de Metro desde CSV.
"""

#Libreriass
import pandas as pd
import numpy as np
import geopandas as gpd 
import logging
from pathlib import Path
import sys
from io import StringIO
import unicodedata
import re

# --- Configuración de Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s.%(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger(__name__)

# Definición de rutas y constantes
#rutas de entrada/salida y constantes para el procesamiento de datos
try:
    RUTA_BASE = Path(r"H:\git\proyecto grupal 2\Yelp-Gmaps-Proyecto-DS").resolve()
    RUTA_RAW = RUTA_BASE / "data" / "raw"
    RUTA_PROCESSED = RUTA_BASE / "data" / "processed"
    RUTA_DIM = RUTA_PROCESSED / "dim" # Carpeta para dimensiones si se crearan

    # RUTA DE ENTRADA
    RUTA_CSV_SUBWAY = (RUTA_RAW / "estaciones de tren" / "MTA_Subway_Entrances_and_Exits__2024_20250420.csv").resolve()

    # Rutas de SALIDA 
    RUTA_SALIDA_SUBWAY_GPKG = (RUTA_PROCESSED / "subway_entrances_processed.gpkg").resolve()
    RUTA_SALIDA_SUBWAY_CSV = (RUTA_PROCESSED / "subway_entrances_processed.csv").resolve()

    # Creamos los directorios necesarios
    RUTA_PROCESSED.mkdir(parents=True, exist_ok=True)
    RUTA_DIM.mkdir(parents=True, exist_ok=True)

    #  CRS Geográfico estándar 
    CRS_GEOGRAFICO = "EPSG:4326"

    # Mapeamos los códigos de distrito a nombres
    MAPEO_DISTRITOS = {'M': 'Manhattan', 'B': 'Brooklyn', 'Q': 'Queens', 'X': 'Bronx', 'S': 'Staten Island'}


except Exception as e:
    log.exception(f"Error crítico al definir rutas: {e}")
    sys.exit(1)

# creamos un dicionario  con las traducciones de columnas
TRADUCCIONES_COLUMNAS_SUBWAY = {
    'division': 'division_mta',
    'line': 'linea_mta',
    'borough': 'codigo_distrito', 
    'stop_name': 'nombre_parada_complejo', 
    'complex_id': 'id_complejo_estacion',
    'constituent_station_name': 'nombre_estacion_constituyente',
    'station_id': 'id_estacion',
    'gtfs_stop_id': 'id_parada_gtfs',
    'daytime_routes': 'rutas_diurnas',
    'entrance_type': 'tipo_entrada', 
    'entry_allowed': 'entrada_permitida', 
    'exit_allowed': 'salida_permitida', 
    'entrance_latitude': 'latitud_entrada', 
    'entrance_longitude': 'longitud_entrada', 
    'entrance_georeference': 'geometria_wkt' 
}

# --- Funciones Auxiliares ---

#funcion para limpiar nombres de columnas
def limpiar_nombre_columna(col_name: str) -> str:
    if not isinstance(col_name, str): col_name = str(col_name)
    try:
        col_name = col_name.replace('(', ' ').replace(')', ' ').replace('/', '_').replace('-', '_').replace('.', '')
        normalized = unicodedata.normalize('NFKD', col_name).encode('ASCII', 'ignore').decode('utf-8').lower()
        snake_case = re.sub(r'\s+', '_', normalized)
        snake_case = re.sub(r'[^a-z0-9_]+', '', snake_case)
        snake_case = snake_case.strip('_')
        if not snake_case: return f"col_limpia_{hash(col_name)}"
        if snake_case.isdigit(): return f"col_{snake_case}"
        return snake_case
    except Exception as e: log.warning(f"No se pudo limpiar '{col_name}': {e}."); return col_name

# Funciones de Procesamiento para Entradas de Metro de NYC

def cargar_datos_subway(ruta: Path) -> pd.DataFrame | None:
    """Carga datos del CSV de Entradas de Metro (asume header=0)."""
    log.info(f"Intentando cargar datos Subway desde: {ruta}")
    if not ruta.is_file(): log.error(f"Ruta Subway no es archivo: {ruta}"); return None
    df = None
    try:
        dtype_inicial = {
            'Complex ID': str, 'Station ID': str, 'GTFS Stop ID': str,
            'Entrance Latitude': str, 'Entrance Longitude': str
        }
        df = pd.read_csv(ruta, encoding='utf-8', header=0, dtype=dtype_inicial, low_memory=False)
        log.info(f"Archivo Subway '{ruta.name}' cargado con UTF-8 (header=0).")
    except UnicodeDecodeError:
        log.warning(f"Fallo UTF-8 Subway '{ruta.name}'. Intentando Latin-1...")
        try:
            df = pd.read_csv(ruta, encoding='latin1', header=0, dtype=dtype_inicial, low_memory=False)
            log.warning(f"Archivo Subway '{ruta.name}' cargado con Latin-1 (header=0).")
        except Exception as e_latin: log.exception(f"Fallo Subway Latin-1: {e_latin}"); return None
    except Exception as e_main: log.exception(f"Error carga Subway: {e_main}"); return None

    if df is None or df.empty: log.warning("DataFrame Subway vacío/None tras carga."); return df
    log.info(f"Carga Subway: {df.shape[0]} filas, {df.shape[1]} cols.")
    log.debug(f"Columnas originales Subway: {df.columns.tolist()}")
    return df

def transformar_datos_subway(df: pd.DataFrame) -> gpd.GeoDataFrame | pd.DataFrame:
    """Pipeline de transformación para datos de Entradas de Metro."""
    if df is None or df.empty: log.error("Input inválido."); return pd.DataFrame()
    log.info("Inicio transformación Entradas Subway (incluye geoespacial)...")

    # 1. Limpiamos y renombramos las  columnas
    try:
        df.columns = [limpiar_nombre_columna(col) for col in df.columns]
        log.info(f"Columnas Subway limpiadas: {df.columns.tolist()}")
        df.rename(columns=TRADUCCIONES_COLUMNAS_SUBWAY, inplace=True)
        log.info(f"Columnas Subway renombradas: {df.columns.tolist()}")
    except Exception as e: log.exception("Error limpieza/renombrado Subway."); return pd.DataFrame()

    # 2. Funcion para limpieza y Conversión de Tipos
    log.info("Aplicando conversiones de tipo y limpieza Subway...")
    try:
        # Mapear y categorizar Distrito
        if 'codigo_distrito' in df.columns:
            df['nombre_distrito'] = df['codigo_distrito'].map(MAPEO_DISTRITOS)
            df['nombre_distrito'] = df['nombre_distrito'].astype('category')
            # df.drop(columns=['codigo_distrito'], inplace=True) # Opcional: eliminar código si no se necesita
            log.info("Columna 'nombre_distrito' creada y 'codigo_distrito' mapeado.")

        # Booleanos (YES/NO a True/False)
        map_yes_no = {'YES': True, 'NO': False}
        for col_bool in ['entrada_permitida', 'salida_permitida']:
            if col_bool in df.columns:
                df[col_bool] = df[col_bool].astype(str).str.upper().map(map_yes_no).astype('boolean')

        # Categorías
        for col_cat in ['division_mta', 'linea_mta', 'tipo_entrada']:
            if col_cat in df.columns:
                df[col_cat] = df[col_cat].astype('category')

        # pasamos a strings (Limpieza básica)
        for col_str in ['nombre_parada_complejo', 'nombre_estacion_constituyente', 'rutas_diurnas']:
             if col_str in df.columns:
                 df[col_str] = df[col_str].astype('string').str.strip() 

        # IDs como String
        for col_id in ['id_complejo_estacion', 'id_estacion', 'id_parada_gtfs']:
            if col_id in df.columns:
                 df[col_id] = df[col_id].astype('string')

        # Cambismos el formato de las cordenadas a numérico (por si acaso, aunque usaremos WKT)
        for col_coord in ['latitud_entrada', 'longitud_entrada']:
             if col_coord in df.columns:
                 df[col_coord] = pd.to_numeric(df[col_coord], errors='coerce')

    except Exception as e: log.warning(f"Error durante conversiones/limpieza de tipos: {e}")

    # 3. Procesamos la parte Geoespacial
    log.info("Procesando geometría WKT...")
    col_wkt = 'geometria_wkt'; gdf = df # Fallback

    if col_wkt in df.columns:
        try:
            log.info(f"Convirtiendo WKT a geometría con CRS: {CRS_GEOGRAFICO}")
            # Creamos GeoSeries desde WKT
            geometry_col = gpd.GeoSeries.from_wkt(df[col_wkt], crs=CRS_GEOGRAFICO, on_invalid='ignore')
            # Creamos GeoDataFrame solo si hay geometrías válidas
            if not geometry_col.isnull().all():
                gdf = gpd.GeoDataFrame(df, geometry=geometry_col, crs=CRS_GEOGRAFICO)
                # Eliminamos las  columnas redundantes si la conversión fue exitosa
                cols_a_quitar = [col_wkt, 'latitud_entrada', 'longitud_entrada']
                gdf.drop(columns=[col for col in cols_a_quitar if col in gdf.columns], inplace=True)
                log.info("Conversión a GeoDataFrame completada. Columnas WKT/Lat/Lon eliminadas.")
                valid_geom = gdf.geometry.is_valid.sum()
                if valid_geom < len(gdf): log.warning(f"{len(gdf) - valid_geom} geometrías inválidas/ignoradas.")
            else:
                log.warning("Conversión WKT resultó en geometrías nulas. Manteniendo DataFrame.")
                if col_wkt in df.columns: df.drop(columns=[col_wkt], inplace=True) # Eliminar WKT si no sirvió
                gdf = df
        except ImportError: log.error("¡GeoPandas no encontrado!"); gdf = df
        except Exception as e: log.exception("Error convirtiendo WKT."); gdf = df
    else: log.warning(f"Columna WKT '{col_wkt}' no encontrada.");

    log.info("Transformación Subway completada.")
    return gdf

# --- Función de Guardado ---
def guardar_salidas_subway(gdf: gpd.GeoDataFrame | pd.DataFrame, ruta_gpkg: Path, ruta_csv: Path):
    """Guarda el (Geo)DataFrame de Entradas de Metro en GPKG y CSV."""
    log.info("Inicio guardado salidas Subway (GPKG y CSV)...")
    if gdf is None or gdf.empty: log.warning("(Geo)DataFrame Subway vacío."); return

    is_geodataframe = isinstance(gdf, gpd.GeoDataFrame) and 'geometry' in gdf.columns and not gdf['geometry'].isnull().all()

    # Reordenamos las columnas para mejor legibilidad
    cols_ordenadas = [
        'id_objeto_fuente', 
        'id_complejo_estacion', 'id_estacion', 'id_parada_gtfs',
        'nombre_parada_complejo', 'nombre_estacion_constituyente',
        'division_mta', 'linea_mta', 'rutas_diurnas',
        'tipo_entrada', 'entrada_permitida', 'salida_permitida',
        'nombre_distrito', 'codigo_distrito', # Incluir ambos si se mantuvieron
        'geometry' # Geometría al final
    ]
    # Añadimos la latitud y longitud si no se creó geometría
    if not is_geodataframe and 'latitud_entrada' in gdf.columns and 'longitud_entrada' in gdf.columns:
        cols_ordenadas.insert(-1, 'latitud_entrada') # Insertar antes de geometry (que no estará)
        cols_ordenadas.insert(-1, 'longitud_entrada')

    cols_finales = [col for col in cols_ordenadas if col in gdf.columns]
    cols_otras = [col for col in gdf.columns if col not in cols_finales]
    gdf_guardar = gdf[cols_finales + cols_otras]

    # Guardadamos en formato  GPKG
    if is_geodataframe:
        try:
            log.info(f"Guardando Subway GPKG: {ruta_gpkg}")
            gdf_guardar.to_file(ruta_gpkg, driver='GPKG', layer='entradas_subway')
            log.info("Subway GPKG guardado.")
        except ImportError: log.error("¡Falta Fiona/GeoPandas! No se guardó GPKG.")
        except Exception as e_gpkg: log.exception(f"Error guardando Subway GPKG: {e_gpkg}")
    else: log.warning("Omitiendo guardado GPKG (no es GeoDataFrame válido).")

    # Guardar en csv el resultado final
    try:
        log.info(f"Guardando Subway CSV: {ruta_csv}")
        # La columna 'geometry' (si existe) se convierte a WKT
        gdf_guardar.to_csv(ruta_csv, index=False, encoding='utf-8-sig')
        log.info("Subway CSV guardado.")
    except Exception as e_csv: log.exception(f"Error guardando Subway CSV: {e_csv}")


# Función  de ejecucion principal
def main_subway():
    """Orquesta el flujo ETL para datos de Entradas de Metro."""
    log.info("================================================")
    log.info("=== INICIO ETL: Entradas y Salidas de Metro NYC ===")
    log.info("================================================")

    # 0. Validación de entrada
    log.info(f"Verificando entrada Subway: {RUTA_CSV_SUBWAY}")
    if not RUTA_CSV_SUBWAY.is_file(): log.critical(f"Abortando: Archivo Subway no encontrado: {RUTA_CSV_SUBWAY}"); sys.exit(1)
    log.info("Entrada Subway encontrada.")

    # 1. Extracción de datos Subway
    log.info("--- Fase 1: Extracción Subway ---")
    df_raw_subway = cargar_datos_subway(RUTA_CSV_SUBWAY)
    if df_raw_subway is None or df_raw_subway.empty: log.critical("Fallo extracción Subway."); sys.exit(1)

    # 2. Transformación  (incluye geoespacial)
    log.info("--- Fase 2: Transformación Subway ---")
    gdf_processed_subway = transformar_datos_subway(df_raw_subway.copy())
    if gdf_processed_subway.empty: log.critical("Transformación Subway vacía."); sys.exit(1)
    log.info(f"Tipo de dato tras transformación: {type(gdf_processed_subway)}")

    # 3. Carga (Guardado Final GPKG y CSV)
    log.info("--- Fase 3: Carga (Guardado Salidas Subway) ---")
    guardar_salidas_subway(
        gdf_processed_subway,
        RUTA_SALIDA_SUBWAY_GPKG,
        RUTA_SALIDA_SUBWAY_CSV
    )

    log.info("===================================================")
    log.info("=== ETL Entradas de Metro completado ===")
    log.info(f"Salida Principal -> {RUTA_SALIDA_SUBWAY_GPKG.name} Y {RUTA_SALIDA_SUBWAY_CSV.name}")
    log.info("===================================================")

if __name__ == "__main__":
    main_subway()