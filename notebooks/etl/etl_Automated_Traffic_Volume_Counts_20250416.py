# -*- coding: utf-8 -*-
# Script ETL para procesar datos de Solicitudes Históricas "Open Restaurants" NYC
# y actualizar/utilizar la dimensión de nombres de restaurantes, incluyendo
# un indicador de permiso temporal.

# --- Importaciones Esenciales ---
import pandas as pd
import numpy as np
import hashlib # Necesario para generar IDs de nombre
import re
import unicodedata
import logging
from pathlib import Path
import sys
from io import StringIO # Para capturar salida de df.info()

# --- Configuración de Logging Profesional ---
# Configura un logger para seguir el flujo del script y diagnosticar problemas.
logging.basicConfig(
    level=logging.INFO, # Nivel mínimo: INFO (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s [%(levelname)s] %(name)s.%(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger(__name__) # Logger específico para este módulo

# --- Definición de Rutas y Constantes ---
# Gestión centralizada y robusta de rutas usando pathlib.
try:
    # 1. RUTA_BASE: La carpeta raíz PRINCIPAL de tu proyecto.
    #    ¡Asegúrate que esta sea la correcta!
    RUTA_BASE = Path(r"H:\git\proyecto grupal 2\Yelp-Gmaps-Proyecto-DS").resolve()

    # 2. Subcarpetas estándar relativas a RUTA_BASE.
    RUTA_RAW = RUTA_BASE / "data" / "raw"
    RUTA_PROCESSED = RUTA_BASE / "data" / "processed"
    RUTA_DIM = RUTA_PROCESSED / "dim"

    # --- Rutas Específicas para el ETL de Permisos ---

    # 3. Entrada Permisos: Apunta al archivo CSV crudo dentro de la carpeta 'raw'.
    RUTA_CSV_PERMISOS = (RUTA_RAW / "restaurantes temporales" / "Open_Restaurant_Applications__Historic__20250416.csv").resolve()

    # 4. Dimensión Nombres: Apunta al archivo CSV de dimensión en la carpeta 'dim'.
    #    Este archivo se lee y se sobreescribe.
    RUTA_DIM_NOMBRES = (RUTA_DIM / "dim_nombre_restaurante_limpia.csv").resolve()

    # 5. Salida Permisos: Apunta al archivo CSV donde guardaremos el resultado procesado.
    RUTA_SALIDA_PERMISOS_CSV = (RUTA_PROCESSED / "permisos_restaurantes_processed.csv").resolve()

    # --- Creación de Directorios ---
    # Aseguramos que las carpetas de salida existan antes de intentar guardar.
    RUTA_PROCESSED.mkdir(parents=True, exist_ok=True)
    RUTA_DIM.mkdir(parents=True, exist_ok=True)

except Exception as e:
    log.exception(f"Error crítico al definir o resolver rutas de archivo: {e}")
    sys.exit(1) # Salir si hay problemas con las rutas base
# --- Diccionario de Traducción para Permisos ---
# Mapeo explícito de nombres originales limpios (snake_case) a nombres en español.
TRADUCCIONES_COLUMNAS_PERMISOS = {
    'objectid': 'id_objeto',
    'globalid': 'id_global',
    'seating_interest_sidewalkroadwayboth': 'interes_asientos',
    'restaurant_name': 'nombre_restaurante_reportado',
    'legal_business_name': 'nombre_legal_negocio',
    'doing_business_as_dba': 'nombre_dba', # Columna clave para la dimensión
    'building_number': 'numero_edificio',
    'street': 'calle',
    'borough': 'distrito',
    'postcode': 'codigo_postal',
    'business_address': 'direccion_negocio',
    'food_service_establishment_permit': 'permiso_establecimiento_salud',
    'sidewalk_dimensions_length': 'acera_largo',
    'sidewalk_dimensions_width': 'acera_ancho',
    'sidewalk_dimensions_area': 'acera_area',
    'roadway_dimensions_length': 'calzada_largo',
    'roadway_dimensions_width': 'calzada_ancho',
    'roadway_dimensions_area': 'calzada_area',
    'approved_for_sidewalk_seating': 'aprobado_acera',
    'approved_for_roadway_seating': 'aprobado_calzada',
    'qualify_alcohol': 'califica_alcohol',
    'sla_serial_number': 'num_serie_sla',
    'sla_license_type': 'tipo_licencia_sla',
    'landmark_district_or_building': 'es_landmark',
    'landmarkdistrict_terms': 'terminos_landmark', # Se eliminará
    'healthcompliance_terms': 'terminos_salud',   # Se eliminará
    'time_of_submission': 'fecha_solicitud',
    'latitude': 'latitud',
    'longitude': 'longitud', # Se corregirá a longitud después
    'community_board': 'distrito_comunitario',
    'council_district': 'distrito_consejo',
    'census_tract': 'zona_censo',
    'bin': 'bin',
    'bbl': 'bbl',
    'nta': 'nta'
}

# --- Funciones Auxiliares ---

def limpiar_nombre_columna(col_name: str) -> str:
    """Estandariza un nombre de columna a formato snake_case."""
    if not isinstance(col_name, str): col_name = str(col_name)
    try:
        col_name = col_name.replace('(', ' ').replace(')', ' ') # Manejar paréntesis
        normalized = unicodedata.normalize('NFKD', col_name).encode('ASCII', 'ignore').decode('utf-8').lower()
        snake_case = re.sub(r'\s+', '_', normalized)
        snake_case = re.sub(r'[^a-z0-9_]+', '', snake_case)
        snake_case = snake_case.strip('_')
        if not snake_case: log.warning(f"'{col_name}' vacío tras limpieza."); return f"col_limpia_{hash(col_name)}"
        return snake_case
    except Exception as e: log.warning(f"No se pudo limpiar '{col_name}': {e}."); return col_name

def generar_id_nombre(nombre: str) -> str | None:
    """Genera un ID hash corto determinista tipo NOMXXXXXX para el nombre."""
    if pd.isna(nombre) or not str(nombre).strip(): return None # Ignorar nulos y vacíos
    try:
        nombre_std = str(nombre).strip().upper()
        hash_id = hashlib.md5(nombre_std.encode('utf-8')).hexdigest()[:6].upper()
        return f"NOM{hash_id}"
    except Exception as e: log.warning(f"No se pudo generar ID para '{nombre}': {e}."); return None

# --- Funciones de Procesamiento para Permisos ---

def cargar_datos_csv_permisos(ruta: Path) -> pd.DataFrame | None:
    """Carga datos del CSV de Permisos (asume header=0)."""
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

def transformar_datos_permisos(df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline de transformación para datos de Permisos Open Restaurants."""
    if df is None or df.empty: log.error("Input inválido (None o vacío)."); return pd.DataFrame()
    log.info("Inicio transformación Permisos...")

    # 1. Limpiar nombres de columna
    try: df.columns = [limpiar_nombre_columna(col) for col in df.columns]
    except Exception as e: log.exception("Error limpieza nombres Permisos."); return pd.DataFrame()
    log.info(f"Columnas Permisos limpiadas: {df.columns.tolist()}")

    # 2. Renombrar a español (método manual robusto)
    try:
        nuevos_nombres = [TRADUCCIONES_COLUMNAS_PERMISOS.get(col, col) for col in df.columns]
        renombradas_count = sum(1 for old, new in zip(df.columns, nuevos_nombres) if old != new)
        df.columns = nuevos_nombres
        if renombradas_count == 0: log.warning("Ninguna columna Permisos renombrada.")
        else: log.info(f"{renombradas_count} cols Permisos renombradas.")
        log.info(f"Nombres finales Permisos: {df.columns.tolist()}")
    except Exception as e: log.exception("Error renombrado manual Permisos."); return pd.DataFrame()

    # 3. Conversión de tipos y limpieza específica
    log.info("Aplicando conversiones de tipo y limpieza Permisos...")
    # Fechas
    col_fecha = 'fecha_solicitud';
    if col_fecha in df.columns:
        try: df[col_fecha] = pd.to_datetime(df[col_fecha], errors='coerce')
        except Exception as e: log.warning(f"Error convirtiendo '{col_fecha}': {e}")
    # Numéricos
    cols_numericas = ['acera_largo', 'acera_ancho', 'acera_area', 'calzada_largo', 'calzada_ancho', 'calzada_area', 'latitud', 'longitude']
    if 'longitude' in df.columns: df.rename(columns={'longitude':'longitud'}, inplace=True)
    for col in cols_numericas:
        if col in df.columns:
            try: df[col] = pd.to_numeric(df[col], errors='coerce')
            except Exception as e: log.warning(f"Error convirtiendo '{col}' a numérico: {e}")
    # Booleanos/Categóricos (Indicadores Yes/No)
    cols_bool_like = ['aprobado_acera', 'aprobado_calzada', 'califica_alcohol', 'es_landmark']
    mapa_si_no = {'yes': True, 'no': False, 'true': True, 'false': False}
    for col in cols_bool_like:
        if col in df.columns:
            try: df[col] = df[col].astype(str).str.lower().str.strip().map(mapa_si_no).astype('boolean')
            except Exception as e: log.warning(f"Error convirtiendo '{col}' a booleano: {e}")
    # Strings y Códigos
    cols_string = [
        'id_global', 'interes_asientos', 'nombre_restaurante_reportado', 'nombre_legal_negocio',
        'nombre_dba', 'numero_edificio', 'calle', 'distrito', 'codigo_postal', 'direccion_negocio',
        'permiso_establecimiento_salud', 'num_serie_sla', 'tipo_licencia_sla', 'terminos_landmark',
        'terminos_salud', 'distrito_comunitario', 'distrito_consejo', 'zona_censo', 'bin', 'bbl', 'nta']
    for col in cols_string:
        if col in df.columns:
            try:
                df[col] = df[col].astype(str).str.strip().replace(['nan', 'None', 'undefined'], '', regex=False)
                df[col] = df[col].astype('string')
            except Exception as e: log.warning(f"Error limpiando/convirtiendo '{col}' a string: {e}")
    # Estandarizar Distrito
    col_distrito = 'distrito'
    if col_distrito in df.columns: df[col_distrito] = df[col_distrito].astype(str).str.title().astype('category')
    # Manejo Lat/Lon cero
    col_lat = 'latitud'; col_lon = 'longitud'
    if col_lat in df.columns: df[col_lat] = df[col_lat].replace(0.0, np.nan)
    if col_lon in df.columns: df[col_lon] = df[col_lon].replace(0.0, np.nan)

    # 4. Añadir columna 'tipo_operacion'
    log.info("Añadiendo columna 'tipo_operacion' = 'Temporal_Exterior'...")
    df['tipo_operacion'] = 'Temporal_Exterior'
    df['tipo_operacion'] = df['tipo_operacion'].astype('category')

    # 5. Eliminar columnas innecesarias
    cols_a_eliminar = ['terminos_landmark', 'terminos_salud'] # Columnas identificadas como internas/redundantes
    cols_existentes_a_eliminar = [col for col in cols_a_eliminar if col in df.columns]
    if cols_existentes_a_eliminar: df.drop(columns=cols_existentes_a_eliminar, inplace=True)
    log.info(f"Columnas eliminadas: {cols_existentes_a_eliminar}")

    log.info("Transformación Permisos completada.")
    return df

# --- Función de Gestión de Dimensión Nombres con Flag ---
def gestionar_dimension_nombres_con_flag(
    df_fuente: pd.DataFrame,
    ruta_dim: Path,
    col_nombre_fuente: str,
    flag_permiso_temporal: bool = False
) -> pd.DataFrame:
    """
    Gestiona la dimensión de nombres: carga, inicializa/actualiza flag 'permiso_temporal',
    y añade nombres nuevos desde df_fuente con el flag especificado.

    Args:
        df_fuente (pd.DataFrame): DataFrame con los nombres a verificar/añadir.
        ruta_dim (Path): Ruta al archivo CSV de la dimensión de nombres.
        col_nombre_fuente (str): Nombre de la columna en df_fuente con los nombres.
        flag_permiso_temporal (bool): Indica si los nombres de esta fuente deben
                                       marcarse/actualizarse como con permiso temporal.

    Returns:
        pd.DataFrame: La dimensión de nombres actualizada y lista para guardar.
    """
    log.info(f"Inicio gestión dimensión nombres desde '{col_nombre_fuente}'. Flag Temporal={flag_permiso_temporal}.")
    col_flag = 'permiso_temporal' # Nombre de la columna a añadir/actualizar

    # --- Carga e Inicialización de Dimensión ---
    dim_nombres = pd.DataFrame(columns=['id_nombre', 'nombre', col_flag])
    if ruta_dim.is_file():
        try:
            # Cargar especificando tipos base para robustez
            dim_nombres = pd.read_csv(ruta_dim, dtype={'id_nombre': str, 'nombre': str}, keep_default_na=False) # keep_default_na=False para leer booleanos correctamente
            log.info(f"Dimensión cargada desde '{ruta_dim.name}': {dim_nombres.shape[0]} registros.")
            # Inicializar columna flag si no existe, con default False
            if col_flag not in dim_nombres.columns:
                log.warning(f"'{col_flag}' no encontrada. Añadiendo con default=False.")
                dim_nombres[col_flag] = False
            # Asegurar tipo booleano nullable y manejar NaNs o valores no booleanos previos
            dim_nombres[col_flag] = pd.to_numeric(dim_nombres[col_flag], errors='ignore') # Intentar convertir números
            map_bool = {'true': True, 'false': False, '1': True, '0': False, '1.0': True, '0.0': False,
                        True: True, False: False, np.nan: False, None: False, '':False}
            dim_nombres[col_flag] = dim_nombres[col_flag].astype(str).str.lower().map(map_bool).fillna(False).astype('boolean')

            if not all(c in dim_nombres.columns for c in ['id_nombre', 'nombre', col_flag]):
                 log.error("Dimensión cargada inválida. Se reseteará."); dim_nombres = pd.DataFrame(columns=['id_nombre', 'nombre', col_flag])
        except Exception as e: log.exception(f"Error cargando/inicializando dimensión. Se usará vacía."); dim_nombres = pd.DataFrame(columns=['id_nombre', 'nombre', col_flag])
    else: log.warning(f"Dimensión '{ruta_dim.name}' no encontrada. Se creará nueva.")

    # Asegurar tipo booleano antes de procesar
    if col_flag not in dim_nombres.columns: dim_nombres[col_flag] = False
    dim_nombres[col_flag] = dim_nombres[col_flag].fillna(False).astype('boolean')

    # --- Procesamiento ---
    if col_nombre_fuente not in df_fuente.columns:
        log.error(f"Columna '{col_nombre_fuente}' ausente en df_fuente."); return dim_nombres

    # Usar un set para eficiencia, asegurándose de que no sean nulos o vacíos
    nombres_fuente_unicos = set(n for n in df_fuente[col_nombre_fuente].dropna().unique() if str(n).strip())
    log.info(f"{len(nombres_fuente_unicos)} nombres únicos válidos en datos fuente.")
    if not nombres_fuente_unicos: log.warning("No hay nombres válidos en fuente."); return dim_nombres

    # --- 1. Actualizar Flag para Nombres Existentes (SOLO si flag_permiso_temporal es True) ---
    if flag_permiso_temporal:
        nombres_existentes_en_dim = set(dim_nombres['nombre'])
        nombres_a_actualizar = nombres_fuente_unicos.intersection(nombres_existentes_en_dim)
        if nombres_a_actualizar:
            log.info(f"Actualizando '{col_flag}'=True para {len(nombres_a_actualizar)} nombres existentes.")
            mascara_actualizar = dim_nombres['nombre'].isin(nombres_a_actualizar)
            # Actualizar solo los que actualmente son False a True
            # dim_nombres.loc[mascara_actualizar & (dim_nombres[col_flag] == False), col_flag] = True
            # O, más simple, si aparece en esta fuente, se marca como True independientemente del valor anterior:
            dim_nombres.loc[mascara_actualizar, col_flag] = True
        else:
            log.info("No hay nombres existentes para actualizar flag.")

    # --- 2. Identificar y Añadir Nombres Nuevos ---
    nombres_existentes_en_dim = set(dim_nombres['nombre']) # Recalcular
    nombres_nuevos = nombres_fuente_unicos - nombres_existentes_en_dim

    if not nombres_nuevos:
        log.info("No se encontraron nuevos nombres para agregar."); return dim_nombres # Retornar la actualizada

    log.info(f"Identificados {len(nombres_nuevos)} nuevos nombres. Generando IDs...")
    nuevos_registros = []
    for nombre in nombres_nuevos:
        id_gen = generar_id_nombre(nombre)
        if id_gen:
            nuevos_registros.append({
                'id_nombre': id_gen, 'nombre': nombre, col_flag: flag_permiso_temporal
            })

    if not nuevos_registros: log.warning("No se generaron registros nuevos válidos."); return dim_nombres

    df_nuevos = pd.DataFrame(nuevos_registros).drop_duplicates(subset=['nombre'])
    # Chequeo de colisión
    id_counts = df_nuevos['id_nombre'].value_counts(); colisiones = id_counts[id_counts > 1]
    if not colisiones.empty: log.warning(f"¡Colisión HASH IDs nuevos: {colisiones.index.tolist()}!")
    df_nuevos = df_nuevos.drop_duplicates(subset=['id_nombre'], keep='first')
    df_nuevos[col_flag] = df_nuevos[col_flag].astype('boolean') # Asegurar tipo

    # Concatenar y limpiar duplicados finales por nombre
    dim_actualizada = pd.concat([dim_nombres, df_nuevos], ignore_index=True)
    dim_actualizada = dim_actualizada.drop_duplicates(subset=['nombre'], keep='last')
    # Asegurar tipo final de la columna flag
    dim_actualizada[col_flag] = dim_actualizada[col_flag].fillna(False).astype('boolean')

    log.info(f"Dimensión actualizada a {dim_actualizada.shape[0]} registros.")
    return dim_actualizada


def aplicar_clave_surrogada(df: pd.DataFrame, dim_nombres: pd.DataFrame, col_nombre_origen: str) -> pd.DataFrame:
    """Aplica el id_nombre desde la dimensión al DataFrame principal."""
    log.info(f"Aplicando clave subrogada 'id_nombre' desde '{col_nombre_origen}'...")
    col_id = 'id_nombre'
    if col_nombre_origen not in df.columns: log.error(f"Falta '{col_nombre_origen}'."); return df
    if dim_nombres.empty or not all(c in dim_nombres.columns for c in ['nombre', col_id]): log.error("Dimensión inválida."); return df

    # Crear mapa asegurando unicidad de nombres como índice
    mapa_id = dim_nombres.drop_duplicates(subset=['nombre'], keep='last').set_index('nombre')[col_id]
    df[col_id] = df[col_nombre_origen].map(mapa_id) # Aplicar
    log.info(f"Columna '{col_id}' aplicada/actualizada.")
    if not df.empty: # Calcular estadísticas solo si hay datos
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
        # Definir orden preferido de columnas
        columnas_ordenadas = [
            'id_objeto', 'id_global', 'id_nombre', 'nombre_dba', 'nombre_restaurante_reportado',
            'nombre_legal_negocio', 'permiso_establecimiento_salud', 'tipo_operacion',
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


def guardar_dimension(dim_df: pd.DataFrame, ruta_dim: Path):
    """Guarda el DataFrame de dimensión como CSV."""
    log.info(f"Inicio guardado dimensión en: {ruta_dim}")
    # Asegurarse de que la columna flag exista antes de guardar
    if 'permiso_temporal' not in dim_df.columns:
        log.warning("Falta columna 'permiso_temporal' en dimensión a guardar. Añadiendo con False.")
        dim_df['permiso_temporal'] = False
    # Asegurar tipo antes de guardar
    dim_df['permiso_temporal'] = dim_df['permiso_temporal'].astype('boolean')

    if dim_df is None or dim_df.empty: log.warning("DataFrame dimensión vacío, no se guarda."); return
    try:
        ruta_dim.parent.mkdir(parents=True, exist_ok=True)
        # Reordenar columnas de la dimensión para claridad
        dim_cols_ordered = ['id_nombre', 'nombre', 'permiso_temporal']
        dim_df_save = dim_df[[col for col in dim_cols_ordered if col in dim_df.columns]]

        dim_df_save.to_csv(ruta_dim, index=False, encoding='utf-8-sig')
        log.info(f"Dimensión guardada en: {ruta_dim}")
    except Exception as e: log.exception(f"Error CRÍTICO guardando dimensión en '{ruta_dim}': {e}")


# --- Flujo Principal de Ejecución para Permisos ---
def main_permisos():
    """Orquesta el flujo ETL para Permisos, actualizando dimensión nombres con flag."""
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
    dim_nombres_actualizada = gestionar_dimension_nombres_con_flag(
        df_fuente=df_transformed,
        ruta_dim=RUTA_DIM_NOMBRES,
        col_nombre_fuente=columna_nombre_clave,
        flag_permiso_temporal=True # <-- Marcamos que esta fuente implica permiso temporal
    )

    # 4. Guardar Dimensión Actualizada
    log.info("--- Fase 4: Guardado Dimensión Nombres Actualizada ---")
    guardar_dimension(dim_nombres_actualizada, RUTA_DIM_NOMBRES) # Sobreescribimos

    # 5. Aplicación de Clave Subrogada al DataFrame de Permisos
    log.info("--- Fase 5: Aplicación Clave Subrogada ---")
    df_final = aplicar_clave_surrogada(df_transformed, dim_nombres_actualizada, columna_nombre_clave)

    # 6. Carga (Guardado Final del DataFrame de Permisos)
    log.info("--- Fase 6: Carga (Guardado Salida Permisos) ---")
    guardar_salida_csv(df_final, RUTA_SALIDA_PERMISOS_CSV)

    log.info("===================================================")
    log.info("=== ETL Permisos Open Restaurants completado ===")
    log.info("===================================================")

    log.info("Info del DataFrame Permisos procesado final:")
    buffer = StringIO(); df_final.info(buf=buffer); log.info(buffer.getvalue())
    log.info("Info de la Dimensión de Nombres actualizada final:")
    buffer_dim = StringIO(); dim_nombres_actualizada.info(buf=buffer_dim); log.info(buffer_dim.getvalue())
    log.info(f"Primeras filas de la dimensión actualizada:\n{dim_nombres_actualizada.head().to_string()}")


if __name__ == "__main__":
    # Punto de entrada: ejecuta el flujo principal para los permisos
    main_permisos()