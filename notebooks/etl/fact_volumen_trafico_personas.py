"""
ETL para datos de tráfico peatonal de NYC. 
"""

#Librerias
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

    # *** RUTA DE ENTRADA para CSV Peatonal ***
    RUTA_CSV_PEATONAL = (RUTA_RAW / "trafico peatonal" / "Bi-Annual_Pedestrian_Counts.csv").resolve()

    # --- Rutas de SALIDA ---
   
    RUTA_SALIDA_DIM_UBICACION_GPKG = (RUTA_DIM / "dim_ubicacion_peatonal.gpkg").resolve()
    RUTA_SALIDA_DIM_UBICACION_CSV = (RUTA_DIM / "dim_ubicacion_peatonal.csv").resolve()
    # Tabla de Hechos de Conteo (CSV)
    RUTA_SALIDA_HECHOS_PEATONAL_CSV = (RUTA_PROCESSED / "fact_conteo_peatonal.csv").resolve()

    # Crear directorios de salida si no existen
    RUTA_PROCESSED.mkdir(parents=True, exist_ok=True)
    RUTA_DIM.mkdir(parents=True, exist_ok=True)

    # CRS a asumir para la geometría WKT
    
    CRS_GEOMETRIA = "EPSG:4326"
    # CRS_GEOMETRIA = "EPSG:2263"

except Exception as e:
    log.exception(f"Error crítico al definir rutas: {e}")
    sys.exit(1)

# Creamos dicionarios nesesarios
TRADUCCIONES_COLUMNAS_PEATONAL_ID = {
    'objectid': 'id_objeto_fuente', 
    'loc': 'id_ubicacion',       
    'borough': 'distrito',
    'street_nam': 'calle',
    'from_stree': 'calle_desde',
    'to_street': 'calle_hasta',
    'iex': 'interseccion_excluida', 
    'the_geom': 'geometria_wkt'     
}

# --- Funciones Auxiliares ---
"""Estandarizamos el nombre de la columna snake_case."""
def limpiar_nombre_columna(col_name: str) -> str:
   
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

# --- Funciones de Procesamiento para Peatonal ---

"""Cargamo s los datos de conteo peatonal desde CSV."""
def cargar_datos_peatonal(ruta: Path) -> pd.DataFrame | None:
    
    log.info(f"Intentando cargar datos Peatonal desde: {ruta}")
    if not ruta.is_file(): log.error(f"Ruta Peatonal no es archivo: {ruta}"); return None
    df = None
    try:
        df = pd.read_csv(ruta, encoding='utf-8', header=0, low_memory=False)
        log.info(f"Archivo Peatonal '{ruta.name}' cargado con UTF-8 (header=0).")
    except UnicodeDecodeError:
        log.warning(f"Fallo UTF-8 Peatonal '{ruta.name}'. Intentando Latin-1...")
        try:
            df = pd.read_csv(ruta, encoding='latin1', header=0, low_memory=False)
            log.warning(f"Archivo Peatonal '{ruta.name}' cargado con Latin-1 (header=0).")
        except Exception as e_latin: log.exception(f"Fallo Peatonal Latin-1: {e_latin}"); return None
    except Exception as e_main: log.exception(f"Error carga Peatonal: {e_main}"); return None

    if df is None or df.empty: log.warning("DataFrame Peatonal vacío/None tras carga."); return df
    log.info(f"Carga Peatonal: {df.shape[0]} filas, {df.shape[1]} columnas.")
    log.debug(f"Columnas originales Peatonal: {df.columns.tolist()}")
    return df
"""
    Aplicamos las tmaciones iniciales a datos peatonales: limpia/renombra ID cols,
    procesa geometría WKT a GeoDataFrame. Mantiene columnas de conteo para reshape posterior.

 """
def transformar_inicial_peatonal(df: pd.DataFrame) -> gpd.GeoDataFrame | pd.DataFrame:
  
    if df is None or df.empty: log.error("Input inválido."); return pd.DataFrame()
    log.info("Inicio transformación inicial Peatonal (incluye geoespacial)...")

    original_cols = df.columns.tolist()
    cleaned_cols_map = {orig: limpiar_nombre_columna(orig) for orig in original_cols}
    df.columns = cleaned_cols_map.values()
    log.info("Columnas Peatonal limpiadas (snake_case).")

    # Identificar columnas de conteo (ej. 'may07_am', 'oct23_md', etc.)
    # Asumimos que empiezan con abreviatura de mes (3 letras) seguida de año (2 dígitos) y _AM/_PM/_MD
    patron_conteo = r'^(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\d{2}_(am|pm|md)$'
    cols_conteo = [col for col in df.columns if re.match(patron_conteo, col)]
    log.info(f"Identificadas {len(cols_conteo)} columnas de conteo (ej: {cols_conteo[:3]}...).")

    # Columnas de identificación (las que NO son de conteo)
    cols_id_limpias = [col for col in df.columns if col not in cols_conteo]
    log.info(f"Columnas de identificación: {cols_id_limpias}")

    # Renombrar solo las columnas de identificación a español
    traducciones_actuales = {col_limpia: TRADUCCIONES_COLUMNAS_PEATONAL_ID.get(col_limpia, col_limpia)
                              for col_limpia in cols_id_limpias}
    df.rename(columns=traducciones_actuales, inplace=True)
    log.info(f"Columnas de identificación renombradas: {df.columns.tolist()}") # Mostrar todos los nombres ahora

    # Procesar Geometría
    log.info("Procesando geometría WKT...")
    col_wkt = 'geometria_wkt'; gdf = df # Fallback a DataFrame normal
    if col_wkt in df.columns:
        try:
            log.info(f"Convirtiendo WKT a geometría con CRS: {CRS_GEOMETRIA}")
            # Intentar convertir WKT, manejar geometrías vacías o inválidas
            geometry_col = gpd.GeoSeries.from_wkt(df[col_wkt], crs=CRS_GEOMETRIA, on_invalid='ignore')
            # Crear GeoDataFrame solo si la conversión tuvo éxito y no son todos nulos
            if not geometry_col.isnull().all():
                gdf = gpd.GeoDataFrame(df, geometry=geometry_col, crs=CRS_GEOMETRIA)
                gdf = gdf.drop(columns=[col_wkt]); log.info("Conversión a GeoDataFrame OK.")
                valid_geom = gdf.geometry.is_valid.sum()
                if valid_geom < len(gdf): log.warning(f"{len(gdf) - valid_geom} geometrías inválidas/ignoradas.")
            else:
                log.warning("Conversión WKT resultó en todas geometrías nulas. Manteniendo como DataFrame.")
                if col_wkt in df.columns: df.drop(columns=[col_wkt], inplace=True) # Eliminar WKT si no sirvió
                gdf = df
        except ImportError: log.error("¡GeoPandas no encontrado!"); gdf = df
        except Exception as e: log.exception("Error convirtiendo WKT."); gdf = df
    else: log.warning(f"Columna WKT '{col_wkt}' no encontrada.");

    # Limpieza básica de columnas de identificación
    try:
        if 'id_ubicacion' in gdf.columns: gdf['id_ubicacion'] = gdf['id_ubicacion'].astype(str) # Clave principal como string
        if 'distrito' in gdf.columns: gdf['distrito'] = gdf['distrito'].astype('category')
        for col in ['calle', 'calle_desde', 'calle_hasta']:
             if col in gdf.columns: gdf[col] = gdf[col].astype('string').str.strip() # Nullable string
        # Convertirmos a categorías si no son nulos
        col_iex = 'interseccion_excluida'
        if col_iex in gdf.columns:
            gdf[col_iex] = gdf[col_iex].astype(str).str.lower().map({'y':True, 'n':False}).astype('boolean')

    except Exception as e: log.warning(f"Error en limpieza de IDs/texto: {e}")

    log.info("Transformación inicial Peatonal completada.")
    return gdf # Devolver GeoDataFrame o DataFrame

def crear_y_guardar_dimension_ubicacion(
    gdf_peatonal: gpd.GeoDataFrame | pd.DataFrame,
    ruta_salida_gpkg: Path,
    ruta_salida_csv: Path
):
    log.info("Inicio creación dimensión ubicaciones peatonales...")
    # Columnas que definen una ubicación única
    cols_ubicacion_base = ['id_ubicacion', 'calle', 'calle_desde', 'calle_hasta', 'distrito', 'interseccion_excluida']
    cols_ubicacion = cols_ubicacion_base.copy()
    is_geodataframe = False
    if 'geometry' in gdf_peatonal.columns and isinstance(gdf_peatonal, gpd.GeoDataFrame) and not gdf_peatonal['geometry'].isnull().all():
        cols_ubicacion.append('geometry'); is_geodataframe = True
        log.info("Incluyendo 'geometry' en dimensión ubicación.")
    else: log.warning("Dimensión ubicación se creará sin geometría válida.")

    if not all(col in gdf_peatonal.columns for col in cols_ubicacion_base):
        log.error(f"Faltan columnas base para dim_ubicacion."); return

    try:
        dim_ubicacion = gdf_peatonal[cols_ubicacion].copy()
        log.info(f"Registros ubicación antes de duplicados: {len(dim_ubicacion)}")
        if not dim_ubicacion.dropna(subset=['id_ubicacion'])['id_ubicacion'].is_unique:
             log.warning("'id_ubicacion' no es único. Se mantendrá primera aparición.")
        dim_ubicacion.drop_duplicates(subset=['id_ubicacion'], keep='first', inplace=True, ignore_index=True)
        log.info(f"Dimensión ubicación creada: {len(dim_ubicacion)} ubicaciones únicas.")


        dim_ubicacion['id_ubicacion'] = dim_ubicacion['id_ubicacion'].astype('string')
    

       #ordenamos las columnas para guardar
        cols_ordenadas = ['id_ubicacion', 'calle', 'calle_desde', 'calle_hasta', 'distrito', 'interseccion_excluida', 'geometry']
        cols_finales_dim = [col for col in cols_ordenadas if col in dim_ubicacion.columns]
        cols_otras_dim = [col for col in dim_ubicacion.columns if col not in cols_finales_dim]
        dim_ubicacion_guardar = dim_ubicacion[cols_finales_dim + cols_otras_dim]

        # --- Guardamosn ambosformatos
        if not dim_ubicacion_guardar.empty:
            if is_geodataframe:
                try:
                    log.info(f"Guardando dimensión ubicación GPKG: {ruta_salida_gpkg}")
                    dim_ubicacion_guardar.to_file(ruta_salida_gpkg, driver='GPKG', layer='ubicaciones_peatonal')
                    log.info("Dimensión ubicación GPKG guardada.")
                except ImportError: log.error("¡Falta Fiona/GeoPandas! No se guardó GPKG.")
                except Exception as e_gpkg: log.exception(f"Error guardando GPKG: {e_gpkg}")
            try:
                log.info(f"Guardando dimensión ubicación CSV: {ruta_salida_csv}")
                dim_ubicacion_guardar.to_csv(ruta_salida_csv, index=False, encoding='utf-8-sig')
                log.info("Dimensión ubicación CSV guardada.")
            except Exception as e_csv: log.exception(f"Error guardando CSV: {e_csv}")
        else: log.warning("Dimensión ubicación vacía, no se guardaron archivos.")
    except Exception as e: log.exception(f"Error creando dimensión ubicación: {e}")


def reestructurar_y_guardar_hechos_peatonal(
    df_transformado: pd.DataFrame | gpd.GeoDataFrame,
    ruta_salida_csv: Path
):
    """
    Reestructuramos  los dattos                                                

    """
    log.info("Inicio reestructuración a formato largo y guardado de hechos peatonales...")

    # Identificar columnas de ID (las que NO son de conteo y NO son geometría)
    patron_conteo = r'^(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\d{2}_(am|pm|md)$'
    cols_conteo = [col for col in df_transformado.columns if re.match(patron_conteo, col)]
    cols_id = [col for col in df_transformado.columns if col not in cols_conteo and col != 'geometry']

    if not cols_id or not cols_conteo:
        log.error("No se pudieron identificar columnas de ID o de conteo para reestructurar.")
        return

    log.info(f"Columnas ID para melt: {cols_id}")
    log.info(f"Columnas de Valor para melt: {len(cols_conteo)}")

    try:
        # Reestructurar de ancho a largo
        df_largo = pd.melt(
            df_transformado,
            id_vars=cols_id,
            value_vars=cols_conteo,
            var_name='periodo_conteo_raw', # Nombre temporal
            value_name='conteo_peatonal'
        )
        log.info(f"DataFrame reestructurado a formato largo: {df_largo.shape[0]} filas.")

        
        log.info("Procesando columna 'periodo_conteo_raw'...")
        # Extraer Mes (3 letras), Año (2 dígitos), Periodo (AM/PM/MD)
        patron_parseo = r'([a-z]{3})(\d{2})_(am|pm|md)'
        extracted_data = df_largo['periodo_conteo_raw'].str.extract(patron_parseo, flags=re.IGNORECASE)
        extracted_data.columns = ['mes_abr', 'anio_corto', 'periodo']

       
        map_mes = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                   'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
        extracted_data['mes'] = extracted_data['mes_abr'].str.lower().map(map_mes)

     
        extracted_data['anio'] = pd.to_numeric(extracted_data['anio_corto'], errors='coerce') + 2000

       
        df_largo = pd.concat([df_largo, extracted_data[['anio', 'mes', 'periodo']]], axis=1)
        df_largo.drop(columns=['periodo_conteo_raw', 'anio_corto', 'mes_abr'], inplace=True, errors='ignore')

        
        df_largo['conteo_peatonal'] = pd.to_numeric(df_largo['conteo_peatonal'], errors='coerce').astype('Int64')

       
        original_rows = len(df_largo)
        df_largo.dropna(subset=['conteo_peatonal'], inplace=True)
        if len(df_largo) < original_rows: log.info(f"Eliminadas {original_rows - len(df_largo)} filas con conteo nulo.")

       
        cols_hechos_final = ['id_ubicacion', 'anio', 'mes', 'periodo', 'conteo_peatonal']
        
        if 'id_objeto_fuente' in df_largo.columns: cols_hechos_final.insert(1, 'id_objeto_fuente')

        df_hechos_final = df_largo[[col for col in cols_hechos_final if col in df_largo.columns]]

        
        df_hechos_final['anio'] = df_hechos_final['anio'].astype(int)
        df_hechos_final['mes'] = df_hechos_final['mes'].astype(int)
        df_hechos_final['periodo'] = df_hechos_final['periodo'].astype('category')

        log.info(f"Tabla de hechos peatonal final creada: {len(df_hechos_final)} filas.")

        
        if not df_hechos_final.empty:
            log.info(f"Guardando tabla de hechos peatonal CSV: {ruta_salida_csv}")
            df_hechos_final.to_csv(ruta_salida_csv, index=False, encoding='utf-8-sig')
            log.info("Tabla de hechos peatonal CSV guardada.")
        else:
            log.warning("Tabla de hechos peatonal vacía, no se guardó.")

    except Exception as e:
        log.exception(f"Error reestructurando o guardando tabla de hechos peatonal: {e}")


# --- Flujo Principal ---
def main_peatonal():
    """Orquesta el flujo ETL para datos Peatonales."""
    log.info("================================================")
    log.info("=== INICIO ETL: Tráfico Peatonal NYC ===")
    log.info("================================================")

    # 0. Validación Entrada
    log.info(f"Verificando entrada Peatonal: {RUTA_CSV_PEATONAL}")
    if not RUTA_CSV_PEATONAL.is_file(): log.critical(f"Abortando: Archivo Peatonal no encontrado: {RUTA_CSV_PEATONAL}"); sys.exit(1)
    log.info("Entrada Peatonal encontrada.")

    # 1. Extracción
    log.info("--- Fase 1: Extracción Peatonal ---")
    df_raw_peatonal = cargar_datos_peatonal(RUTA_CSV_PEATONAL)
    if df_raw_peatonal is None or df_raw_peatonal.empty: log.critical("Fallo extracción Peatonal."); sys.exit(1)

    # 2. Transformación Inicial (incluye geoespacial)
    log.info("--- Fase 2: Transformación Inicial Peatonal ---")
    gdf_processed_inicial = transformar_inicial_peatonal(df_raw_peatonal.copy())
    if gdf_processed_inicial.empty: log.critical("Transformación Inicial Peatonal vacía."); sys.exit(1)
    log.info(f"Tipo de dato tras transformación inicial: {type(gdf_processed_inicial)}")

    # 3. Crear y Guardar Dimensión Ubicación (GPKG y CSV)
    log.info("--- Fase 3: Creación/Guardado Dimensión Ubicación Peatonal ---")
    crear_y_guardar_dimension_ubicacion(
        gdf_processed_inicial,
        RUTA_SALIDA_DIM_UBICACION_GPKG,
        RUTA_SALIDA_DIM_UBICACION_CSV
    )

    # 4. Reestructurar a Largo y Guardar Tabla de Hechos
    log.info("--- Fase 4: Reestructurar a Largo y Guardar Hechos Peatonales ---")
    reestructurar_y_guardar_hechos_peatonal(
        gdf_processed_inicial, # Pasamos el resultado de la transformación inicial
        RUTA_SALIDA_HECHOS_PEATONAL_CSV
    )

    log.info("===================================================")
    log.info("=== ETL Tráfico Peatonal completado ===")
    log.info(f"Dimensión Ubicación -> {RUTA_SALIDA_DIM_UBICACION_GPKG.name} Y {RUTA_SALIDA_DIM_UBICACION_CSV.name}")
    log.info(f"Tabla de Hechos -> {RUTA_SALIDA_HECHOS_PEATONAL_CSV.name}")
    log.info("===================================================")


if __name__ == "__main__":
    main_peatonal()