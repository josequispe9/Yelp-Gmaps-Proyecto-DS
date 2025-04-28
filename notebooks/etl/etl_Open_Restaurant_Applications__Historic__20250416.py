

# Libreriasun 
import pandas as pd
import numpy as np
import hashlib
import re
import unicodedata
import logging
from pathlib import Path
import sys
from io import StringIO

# --- Configuración de Logging Profesional ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s.%(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger(__name__)

# --- Definimos las rutas y constantes ---
try:
    RUTA_BASE = Path(r"H:\git\proyecto grupal 2\Yelp-Gmaps-Proyecto-DS")
    RUTA_RAW = RUTA_BASE / "data" / "raw"
    RUTA_PROCESSED = RUTA_BASE / "data" / "processed"
    RUTA_DIM = RUTA_PROCESSED / "dim"

    # Ruta de entrada Permisos
    RUTA_CSV_PERMISOS = (RUTA_RAW / "restaurantes temporales" / "Open_Restaurant_Applications__Historic__20250416.csv").resolve()

    # Ruta Dimensión Nombres (Lectura y Escritura)
    RUTA_DIM_NOMBRES = (RUTA_DIM / "dim_nombre_restaurante_limpia.csv").resolve()

    # Ruta Salida Permisos Procesados
    RUTA_SALIDA_PERMISOS_CSV = (RUTA_PROCESSED / "permisos_restaurantes_processed.csv").resolve()

    # Crear directorios
    RUTA_PROCESSED.mkdir(parents=True, exist_ok=True)
    RUTA_DIM.mkdir(parents=True, exist_ok=True)

except Exception as e:
    log.exception(f"Error crítico al definir rutas: {e}")
    sys.exit(1)

# --- Diccionario de nombres de columnas traducidas a españñol ---
TRADUCCIONES_COLUMNAS_PERMISOS = {
    'objectid': 'id_objeto', 'globalid': 'id_global',
    'seating_interest_sidewalkroadwayboth': 'interes_asientos',
    'restaurant_name': 'nombre_restaurante_reportado',
    'legal_business_name': 'nombre_legal_negocio',
    'doing_business_as_dba': 'nombre_dba', 
    'building_number': 'numero_edificio', 'street': 'calle', 'borough': 'distrito',
    'postcode': 'codigo_postal', 'business_address': 'direccion_negocio',
    'food_service_establishment_permit': 'permiso_establecimiento_salud',
    'sidewalk_dimensions_length': 'acera_largo', 'sidewalk_dimensions_width': 'acera_ancho',
    'sidewalk_dimensions_area': 'acera_area', 'roadway_dimensions_length': 'calzada_largo',
    'roadway_dimensions_width': 'calzada_ancho', 'roadway_dimensions_area': 'calzada_area',
    'approved_for_sidewalk_seating': 'aprobado_acera', 'approved_for_roadway_seating': 'aprobado_calzada',
    'qualify_alcohol': 'califica_alcohol', 'sla_serial_number': 'num_serie_sla',
    'sla_license_type': 'tipo_licencia_sla', 'landmark_district_or_building': 'es_landmark',
    'landmarkdistrict_terms': 'terminos_landmark', 'healthcompliance_terms': 'terminos_salud',
    'time_of_submission': 'fecha_solicitud', 'latitude': 'latitud', 'longitude': 'longitud',
    'community_board': 'distrito_comunitario', 'council_district': 'distrito_consejo',
    'census_tract': 'zona_censo', 'bin': 'bin', 'bbl': 'bbl', 'nta': 'nta'
}

# --- Funciones Auxiliares ---
 """Estandarizamos los nombr de laS columnas a formato snake_case."""

def limpiar_nombre_columna(col_name: str) -> str:
   
    if not isinstance(col_name, str): col_name = str(col_name)
    try:
        col_name = col_name.replace('(', ' ').replace(')', ' ') # Tratar paréntesis
        normalized = unicodedata.normalize('NFKD', col_name).encode('ASCII', 'ignore').decode('utf-8').lower()
        snake_case = re.sub(r'\s+', '_', normalized)
        snake_case = re.sub(r'[^a-z0-9_]+', '', snake_case)
        snake_case = snake_case.strip('_')
        if not snake_case: log.warning(f"'{col_name}' vacío tras limpieza."); return f"col_limpia_{hash(col_name)}"
        return snake_case
    except Exception as e: log.warning(f"No se pudo limpiar '{col_name}': {e}."); return col_name

"""Generamos un ID para los nombres que no estan en la tabla de dimenciones."""
def generar_id_nombre(nombre: str) -> str | None:
    
    if pd.isna(nombre) or not nombre: return None
    try:
        nombre_std = str(nombre).strip().upper()
        hash_id = hashlib.md5(nombre_std.encode('utf-8')).hexdigest()[:6].upper()
        return f"NOM{hash_id}"
    except Exception as e: log.warning(f"No se pudo generar ID para '{nombre}': {e}."); return None

# --- Funciones de Procesamiento para Permisos ---

def cargar_datos_csv_permisos(ruta: Path) -> pd.DataFrame | None:
    """Carga datos del CSV de Permisos (header=0)."""
    log.info(f"Intentando cargar datos Permisos desde: {ruta}")
    if not ruta.is_file(): log.error(f"Ruta Permisos no es archivo: {ruta}"); return None
    df = None
    try:
        df = pd.read_csv(ruta, encoding='utf-8', header=0, low_memory=False)
        log.info(f"Archivo Permisos '{ruta.name}' cargado con UTF-8 (header=0).")
    except UnicodeDecodeError:
        log.warning(f"Fallo UTF-8 Permisos '{ruta.name}'. Intentando Latin-1...")
        try:
            df = pd.read_csv(ruta, encoding='latin1', header=0, low_memory=False)
            log.warning(f"Archivo Permisos '{ruta.name}' cargado con Latin-1 (header=0).")
        except Exception as e_latin: log.exception(f"Fallo Permisos Latin-1: {e_latin}"); return None
    except Exception as e_main: log.exception(f"Error carga Permisos: {e_main}"); return None

    if df is None or df.empty: log.warning("DataFrame Permisos vacío/None tras carga."); return df
    log.info(f"Carga Permisos: {df.shape[0]} filas, {df.shape[1]} cols.")
    log.debug(f"Columnas originales Permisos: {df.columns.tolist()}")
    return df

"""Pipeline de transformación para datos de Permisos Open Restaurants."""
def transformar_datos_permisos(df: pd.DataFrame) -> pd.DataFrame:

    if df is None or df.empty: log.error("Input inválido (None o vacío)."); return pd.DataFrame()
    log.info("Inicio transformación Permisos...")

    # 1. Limpiamos los nombres de las columnas
    try: df.columns = [limpiar_nombre_columna(col) for col in df.columns]
    except Exception as e: log.exception("Error limpieza nombres Permisos."); return pd.DataFrame()
    log.info(f"Columnas Permisos limpiadas: {df.columns.tolist()}")

    # 2. Renombrarmos las columnas  a español
    try:
        nuevos_nombres = [TRADUCCIONES_COLUMNAS_PERMISOS.get(col, col) for col in df.columns]
        renombradas_count = sum(1 for old, new in zip(df.columns, nuevos_nombres) if old != new)
        df.columns = nuevos_nombres
        if renombradas_count == 0: log.warning("Ninguna col Permisos renombrada.")
        else: log.info(f"{renombradas_count} cols Permisos renombradas.")
        log.info(f"Nombres finales Permisos: {df.columns.tolist()}")
    except Exception as e: log.exception("Error renombrado manual Permisos."); return pd.DataFrame()

    # 3. Conversión de tipos y limpieza específica
    log.info("Aplicando conversiones de tipo y limpieza Permisos...")
    col_fecha = 'fecha_solicitud';
    if col_fecha in df.columns:
        try: df[col_fecha] = pd.to_datetime(df[col_fecha], errors='coerce')
        except Exception as e: log.warning(f"Error convirtiendo '{col_fecha}': {e}")
    cols_numericas = ['acera_largo', 'acera_ancho', 'acera_area', 'calzada_largo', 'calzada_ancho', 'calzada_area', 'latitud', 'longitud']
    if 'longitude' in df.columns: df.rename(columns={'longitude':'longitud'}, inplace=True) # Corregir antes de convertir
    for col in cols_numericas:
        if col in df.columns:
            try: df[col] = pd.to_numeric(df[col], errors='coerce')
            except Exception as e: log.warning(f"Error convirtiendo '{col}' a numérico: {e}")
    cols_bool_like = ['aprobado_acera', 'aprobado_calzada', 'califica_alcohol', 'es_landmark']
    mapa_si_no = {'yes': True, 'no': False, 'true': True, 'false': False}
    for col in cols_bool_like:
        if col in df.columns:
            try:
                df[col] = df[col].astype(str).str.lower().str.strip().map(mapa_si_no).astype('boolean')
            except Exception as e: log.warning(f"Error convirtiendo '{col}' a booleano: {e}")
    cols_string = [
        'id_global', 'interes_asientos', 'nombre_restaurante_reportado', 'nombre_legal_negocio',
        'nombre_dba', 'numero_edificio', 'calle', 'distrito', 'codigo_postal', 'direccion_negocio',
        'permiso_establecimiento_salud', 'num_serie_sla', 'tipo_licencia_sla', 'terminos_landmark',
        'terminos_salud', 'distrito_comunitario', 'distrito_consejo', 'zona_censo', 'bin', 'bbl', 'nta']
    for col in cols_string:
        if col in df.columns:
            try:
                df[col] = df[col].astype(str).str.strip().replace(['nan', 'None', 'undefined'], '', regex=False)
                df[col] = df[col].astype('string') # Usar tipo nullable
            except Exception as e: log.warning(f"Error limpiando/convirtiendo '{col}' a string: {e}")
    col_distrito = 'distrito'
    if col_distrito in df.columns: df[col_distrito] = df[col_distrito].astype(str).str.title().astype('category')
    col_lat = 'latitud'; col_lon = 'longitud'
    if col_lat in df.columns: df[col_lat] = df[col_lat].replace(0.0, np.nan)
    if col_lon in df.columns: df[col_lon] = df[col_lon].replace(0.0, np.nan)

    # 4. Añadimos la columna 'tipo_operacion' si ttiene permiso para trabajar en la vereda ttemporalmente.

    log.info("Añadiendo columna 'tipo_operacion'...")
    df['tipo_operacion'] = 'Temporal_Exterior'
    df['tipo_operacion'] = df['tipo_operacion'].astype('category')

    # 5. Eliminamos columnas innecesarias
    cols_a_eliminar = ['terminos_landmark', 'terminos_salud']
    cols_existentes_a_eliminar = [col for col in cols_a_eliminar if col in df.columns]
    if cols_existentes_a_eliminar: df.drop(columns=cols_existentes_a_eliminar, inplace=True)
    log.info(f"Columnas eliminadas: {cols_existentes_a_eliminar}")

    log.info("Transformación Permisos completada.")
    return df

# --- NUEVA FUNCIÓN DE GESTIÓN DE DIMENSIÓN CON FLAG ---
def gestionar_dimension_nombres_con_flag(
    df_fuente: pd.DataFrame,
    ruta_dim: Path,
    col_nombre_fuente: str,
    flag_permiso_temporal: bool = False 
) -> pd.DataFrame:
    """
    Carga y crea la dimensión de nombres, la inicializa con la columna 'permiso_temporal'
    si no existe (default False), actualiza el flag para nombres existentes encontrados
    en df_fuente (si flag_permiso_temporal=True), y añade nombres nuevos desde df_fuente
    asignándoles el valor de flag_permiso_temporal.

    """
    log.info(f"Inicio gestión dimensión nombres desde '{col_nombre_fuente}'. Flag Temporal={flag_permiso_temporal}.")
    col_flag = 'permiso_temporal' # Nombre de la nueva columna en la dimensión

    # --- Carga e Inicialización de Dimensión ---
    dim_nombres = pd.DataFrame(columns=['id_nombre', 'nombre', col_flag]) # Esquema deseado
    if ruta_dim.is_file():
        try:
            dim_nombres = pd.read_csv(ruta_dim, dtype={'id_nombre': str, 'nombre': str}) # Cargar tipos base
            log.info(f"Dimensión cargada desde '{ruta_dim.name}': {dim_nombres.shape[0]} registros.")
            # Inicializamos columna flag si no existe
            if col_flag not in dim_nombres.columns:
                log.warning(f"Columna '{col_flag}' no encontrada en dimensión. Añadiendo con default=False.")
                dim_nombres[col_flag] = False
           
            map_bool = {'true': True, 'false': False, '1': True, '0': False, '1.0': True, '0.0': False,
                        True: True, False: False, np.nan: False} # Mapeo robusto
            dim_nombres[col_flag] = dim_nombres[col_flag].astype(str).str.lower().map(map_bool).fillna(False).astype('boolean')

            if not all(c in dim_nombres.columns for c in ['id_nombre', 'nombre', col_flag]):
                 log.error("Dimensión cargada inválida. Se reseteará."); dim_nombres = pd.DataFrame(columns=['id_nombre', 'nombre', col_flag])
        except Exception as e: log.exception(f"Error cargando/inicializando dimensión. Se usará vacía."); dim_nombres = pd.DataFrame(columns=['id_nombre', 'nombre', col_flag])
    else: log.warning(f"Dimensión no encontrada '{ruta_dim.name}'. Se creará nueva.")

    # Nos aseguramos el tipo de la columna flag antes de procesar
    if col_flag not in dim_nombres.columns: dim_nombres[col_flag] = False # Asegurar que exista
    dim_nombres[col_flag] = dim_nombres[col_flag].fillna(False).astype('boolean') # Forzar tipo booleano nullable


    # --- Procesamiento ---
    if col_nombre_fuente not in df_fuente.columns:
        log.error(f"Columna '{col_nombre_fuente}' ausente en df_fuente. No se puede actualizar dimensión."); return dim_nombres

    nombres_fuente_unicos = set(df_fuente[col_nombre_fuente].dropna().unique()) # Usar set para eficiencia
    log.info(f"{len(nombres_fuente_unicos)} nombres únicos (no nulos) en datos fuente.")
    if not nombres_fuente_unicos: log.warning("No hay nombres válidos en la fuente para procesar."); return dim_nombres

    # --- 1. Actualizamos Flag para Nombres Existentes (si flag_permiso_temporal es True) ---
    if flag_permiso_temporal:
        nombres_existentes_en_dim = set(dim_nombres['nombre'])
        nombres_a_actualizar = nombres_fuente_unicos.intersection(nombres_existentes_en_dim)
        if nombres_a_actualizar:
            log.info(f"Actualizando flag '{col_flag}'=True para {len(nombres_a_actualizar)} nombres existentes.")
            # Crear máscara booleana para actualizar eficientemente
            mascara_actualizar = dim_nombres['nombre'].isin(nombres_a_actualizar)
            dim_nombres.loc[mascara_actualizar, col_flag] = True
        else:
            log.info("No hay nombres existentes en la dimensión para actualizar el flag temporal.")

    # --- 2. Identificamos y añadimos nombres nuevos  ---
    nombres_existentes_en_dim = set(dim_nombres['nombre']) 
    nombres_nuevos = nombres_fuente_unicos - nombres_existentes_en_dim
    nombres_nuevos = {n for n in nombres_nuevos if n} 

    if not nombres_nuevos:
        log.info("No se encontraron nuevos nombres para agregar."); return dim_nombres # Retornar la actualizada (o no)

    log.info(f"Identificados {len(nombres_nuevos)} nuevos nombres. Generando IDs y registros...")
    nuevos_registros = []
    for nombre in nombres_nuevos:
        id_gen = generar_id_nombre(nombre)
        if id_gen: # Solo añadir si se pudo generar ID
            nuevos_registros.append({
                'id_nombre': id_gen,
                'nombre': nombre,
                col_flag: flag_permiso_temporal # Asignar el flag correspondiente a esta fuente
            })

    if not nuevos_registros:
        log.warning("No se generaron registros válidos para los nombres nuevos."); return dim_nombres

    df_nuevos = pd.DataFrame(nuevos_registros).drop_duplicates(subset=['nombre'])
    id_counts = df_nuevos['id_nombre'].value_counts(); colisiones = id_counts[id_counts > 1]
    if not colisiones.empty: log.warning(f"¡Colisión HASH IDs nuevos: {colisiones.index.tolist()}!")
    df_nuevos = df_nuevos.drop_duplicates(subset=['id_nombre'], keep='first')

    # nOS ASEGURAMOS  que df_nuevos tenga el tipo correcto para la columna flag
    df_nuevos[col_flag] = df_nuevos[col_flag].astype('boolean')

    # ConcatenaNAMOS 
    dim_actualizada = pd.concat([dim_nombres, df_nuevos], ignore_index=True)
    # Limpieza final de datos duplicados por nombre (manteniendo la última aparición)
    dim_actualizada = dim_actualizada.drop_duplicates(subset=['nombre'], keep='last')

    # Nos aseguramos tipo final de la columna flag en toda la dimensión
    dim_actualizada[col_flag] = dim_actualizada[col_flag].fillna(False).astype('boolean')

    log.info(f"Dimensión actualizada a {dim_actualizada.shape[0]} registros.")
    return dim_actualizada

"""Aplica el id_nombre desde la dimensión al DataFrame principal."""
def aplicar_clave_surrogada(df: pd.DataFrame, dim_nombres: pd.DataFrame, col_nombre_origen: str) -> pd.DataFrame:
    
    
    log.info(f"Aplicando clave subrogada 'id_nombre' desde '{col_nombre_origen}'...")
    col_id = 'id_nombre'
    if col_nombre_origen not in df.columns: log.error(f"Falta '{col_nombre_origen}'."); return df
    if dim_nombres.empty or not all(c in dim_nombres.columns for c in ['nombre', col_id]): log.error("Dimensión inválida."); return df
    mapa_id = dim_nombres.drop_duplicates(subset=['nombre'], keep='last').set_index('nombre')[col_id]
    df[col_id] = df[col_nombre_origen].map(mapa_id)
    log.info(f"Columna '{col_id}' aplicada.")
    if not df.empty:
        nulos = df[col_id].isnull().sum(); total = len(df)
        log.info(f"Mapeo ID: {total - nulos}/{total} ({((total - nulos) / total) * 100 if total else 0:.2f}%) OK.")
        if nulos > 0: log.warning(f"{nulos} registros sin ID mapeado.")
    return df

def guardar_salida_csv(df: pd.DataFrame, ruta_salida: Path):
    """Guarda el DataFrame procesado final como CSV."""
    log.info(f"Inicio guardado salida CSV en: {ruta_salida}")
    if df is None or df.empty: log.warning("DataFrame vacío, no se guarda."); return
    try:
        ruta_salida.parent.mkdir(parents=True, exist_ok=True)
        columnas_ordenadas = [
            'id_objeto', 'id_global', 'id_nombre', 'nombre_dba', 'nombre_restaurante_reportado',
            'nombre_legal_negocio', 'permiso_establecimiento_salud', 'tipo_operacion', # Col añadida
            'fecha_solicitud', 'interes_asientos', 'aprobado_acera', 'aprobado_calzada',
            'califica_alcohol', 'num_serie_sla', 'tipo_licencia_sla', 'es_landmark',
            'distrito', 'numero_edificio', 'calle', 'codigo_postal', 'direccion_negocio',
            'latitud', 'longitud', 'distrito_comunitario', 'distrito_consejo',
            'zona_censo', 'nta', 'bin', 'bbl',
            'acera_largo', 'acera_ancho', 'acera_area',
            'calzada_largo', 'calzada_ancho', 'calzada_area'
        ]
        cols_finales = [col for col in columnas_ordenadas if col in df.columns]
        cols_otras = [col for col in df.columns if col not in cols_finales]
        df_guardar = df[cols_finales + cols_otras]
        df_guardar.to_csv(ruta_salida, index=False, encoding='utf-8-sig')
        log.info(f"Datos procesados guardados como CSV en: {ruta_salida}")
    except Exception as e: log.exception(f"Error CRÍTICO guardando CSV en '{ruta_salida}': {e}")

"""Guardamos el DataFrame de dimensión como CSV."""

def guardar_dimension(dim_df: pd.DataFrame, ruta_dim: Path):
    
    log.info(f"Inicio guardado dimensión en: {ruta_dim}")
    if dim_df is None or dim_df.empty: log.warning("DataFrame dimensión vacío, no se guarda."); return
    try:
        ruta_dim.parent.mkdir(parents=True, exist_ok=True)
        dim_df.to_csv(ruta_dim, index=False, encoding='utf-8-sig')
        log.info(f"Dimensión guardada en: {ruta_dim}")
    except Exception as e: log.exception(f"Error CRÍTICO guardando dimensión en '{ruta_dim}': {e}")

# --- Flujo Principal de Ejecución para Permisos (Modificado) ---
"""Orquesta el flujo ETL para Permisos, actualizando dimensión nombres con flag."""

def main_permisos():
    
    log.info("================================================")
    log.info("=== INICIO ETL: Permisos Open Restaurants (con Update Dimensión) ===")
    log.info("================================================")

    # 0. Validación Entrada
    log.info(f"Verificando entrada Permisos: {RUTA_CSV_PERMISOS}")
    if not RUTA_CSV_PERMISOS.is_file(): log.critical(f"Abortando: Archivo Permisos no encontrado: {RUTA_CSV_PERMISOS}"); sys.exit(1)
    log.info("Entrada Permisos encontrada.")

    # 1. Extracción
    log.info("--- Fase 1: Extracción Permisos ---")
    df_raw_permisos = cargar_datos_csv_permisos(RUTA_CSV_PERMISOS)
    if df_raw_permisos is None or df_raw_permisos.empty: log.critical("Fallo extracción Permisos."); sys.exit(1)

    # 2. Transformación
    log.info("--- Fase 2: Transformación Permisos ---")
    df_transformed = transformar_datos_permisos(df_raw_permisos.copy())
    if df_transformed.empty: log.critical("Transformación Permisos vacía."); sys.exit(1)

    # 3. Gestión de Dimensión Nombres (Actualizar existentes y añadir nuevos con flag=True)
    log.info("--- Fase 3: Gestión Dimensión Nombres (con Flag Temporal) ---")
    columna_nombre_clave = 'nombre_dba' # Columna a usar de df_transformed
    # Llamamos a la nueva función indicando que esta fuente SÍ implica permiso temporal
    dim_nombres_actualizada = gestionar_dimension_nombres_con_flag(
        df_fuente=df_transformed,
        ruta_dim=RUTA_DIM_NOMBRES,
        col_nombre_fuente=columna_nombre_clave,
        flag_permiso_temporal=True # <--- ¡Importante! Marcamos como True
    )

    # 4. Guardamos el archivo de dimenciones actualisado
    log.info("--- Fase 4: Guardado Dimensión Nombres ---")
    guardar_dimension(dim_nombres_actualizada, RUTA_DIM_NOMBRES) # Sobreescribimos

    # 5. Aplicación de Clave Subrogada al DataFrame de Permisos
    log.info("--- Fase 5: Aplicación Clave Subrogada ---")
    df_final = aplicar_clave_surrogada(df_transformed, dim_nombres_actualizada, columna_nombre_clave)

    # 6. Carga de log 
    log.info("--- Fase 6: Carga (Guardado Salida Permisos) ---")
    guardar_salida_csv(df_final, RUTA_SALIDA_PERMISOS_CSV)

    log.info("===================================================")
    log.info("=== ETL Permisos Open Restaurants completado ===")
    log.info("===================================================")

    log.info("Info del DataFrame Permisos procesado final:")
    buffer = StringIO(); df_final.info(buf=buffer); log.info(buffer.getvalue())


if __name__ == "__main__":
    main_permisos()