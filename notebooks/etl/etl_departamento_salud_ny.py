import pandas as pd
import numpy as np
import hashlib
import re
import unicodedata
import logging
from pathlib import Path
import sys 

# --- Configuración de Logging ---
# Establece mos un formato estándar para los logs, útil para trazabilidad y depuración.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(module)s.%(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger(__name__)

# --- Definición de Rutas y Constantes ---

""" 
Cargamos los archivos CSV de entrada y salida, así como la dimensión de nombres.
Las rutas deben ser absolutas para evitar problemas de localización.
El uso de Path de pathlib permite una mejor gestión de rutas y compatibilidad entre sistemas operativos.
El uso de try-except asegura que cualquier error en la definición de rutas se maneje adecuadamente.
"""
try:
    # Ruta al archivo CSV fuente (inspecciones). UTILIZAR RUTA ABSOLUTA.
    RUTA_CSV_INSPECCIONES = Path(r"H:\git\proyecto grupal 2\Yelp-Gmaps-Proyecto-DS\data\raw\departamento salubridad NY\DOHMH_New_York_City_Restaurant_Inspection_Results_20250413.csv").resolve()

    # Ruta al archivo CSV de la dimensión de nombres (lectura y escritura).
    RUTA_DIM_NOMBRES = Path(r"H:\git\proyecto grupal 2\Yelp-Gmaps-Proyecto-DS\data\processed\dim\dim_nombre_restaurante_limpia.csv").resolve()
    RUTA_SALIDA_DIM_ACTUALIZADA = RUTA_DIM_NOMBRES # Se sobreescribe el mismo archivo.

    # Ruta de salida para el archivo CSV procesado principal.
    RUTA_SALIDA_PROCESADO_CSV = Path(r"H:\git\proyecto grupal 2\Yelp-Gmaps-Proyecto-DS\data\processed\salubridad_restaurantes_con_ids.csv").resolve()

    # Crear directorios de salida si no existen (mejora la robustez)
    RUTA_DIM_NOMBRES.parent.mkdir(parents=True, exist_ok=True)
    RUTA_SALIDA_PROCESADO_CSV.parent.mkdir(parents=True, exist_ok=True)

except Exception as e:
    log.exception(f"Error al definir o resolver las rutas de archivo: {e}")
    sys.exit(1) # Salir si las rutas base son inválidas
""" 
Realizamos un mapeo explicitto a las colunas para renombrarlas a español.
Esto mejora la legibilidad y mantenibilidad del código, facilitando futuras modificaciones.
El uso de un diccionario permite una fácil modificación y comprensión del esquema resultante.
El uso de nombres en español es preferible para facilitar la comprensión del esquema por parte de los usuarios finales.
"""

TRADUCCIONES_COLUMNAS = {
    'camis': 'id_restaurante', 'dba': 'nombre_restaurante', 'boro': 'distrito',
    'building': 'edificio', 'street': 'calle', 'zipcode': 'codigo_postal',
    'phone': 'telefono', 'cuisine_description': 'tipo_cocina',
    'inspection_date': 'fecha_inspeccion', 'action': 'accion',
    'violation_code': 'codigo_violacion', 'violation_description': 'descripcion_violacion',
    'critical_flag': 'es_critica', 'score': 'puntuacion', 'grade': 'calificacion',
    'grade_date': 'fecha_calificacion', 'record_date': 'fecha_registro',
    'inspection_type': 'tipo_inspeccion', 'latitude': 'latitud', 'longitude': 'longitud',
    'community_board': 'distrito_comunitario', 'council_district': 'distrito_consejo',
    'census_tract': 'zona_censo', 'bin': 'bin', 'bbl': 'bbl', 'nta': 'nta',
    'location_point1': 'punto_ubicacion'
}

# Creamos listas con el tipo de dato  de cada columna para poder definir su formatto .

COLUMNAS_FECHA = ['inspection_date', 'grade_date', 'record_date']
COLUMNAS_NUMERICAS = ['score', 'latitude', 'longitude', 'community_board', 'council_district', 'census_tract', 'bin', 'bbl']
COLUMNAS_CATEGORICAS = ['boro', 'cuisine_description', 'action', 'violation_code', 'violation_description', 'critical_flag', 'grade', 'inspection_type', 'nta']
COLUMNAS_CATEGORICAS_ADICIONALES = ['es_critica', 'calificacion'] 

# --- Funciones de Procesamiento ---
"""
    Estandarizamos un nombre de columna a formato snake_case.
    Normalizamos a ASCII y minúsculas, reemplazamos espacios por guiones bajos,
    y eliminamos caracteres no alfanuméricos.
"""
def limpiar_nombre_columna(col_name):
    
    if not isinstance(col_name, str): col_name = str(col_name)
    try:
        # Normalización robusta a ASCII y minúsculas.
        normalized = unicodedata.normalize('NFKD', col_name).encode('ASCII', 'ignore').decode('utf-8').lower()
        # Reemplazo de espacios y estandarización a snake_case.
        snake_case = re.sub(r'\s+', '_', normalized)
        snake_case = re.sub(r'[^a-z0-9_]+', '', snake_case)
        snake_case = snake_case.strip('_')
        return snake_case
    except Exception as e:
        log.warning(f"No se pudo limpiar el nombre de columna '{col_name}': {e}. Devolviendo original.")
        return col_name 
    
"""
    Generamos un ID corto y determinista (hash MD5 truncado) para un nombre dado.
    Estandariza el nombre (strip, upper) antes de hashear para consistencia.

"""

def generar_id_nombre(nombre):
    if pd.isna(nombre): return None
    try:
        nombre_std = str(nombre).strip().upper()
        hash_id = hashlib.md5(nombre_std.encode('utf-8')).hexdigest()[:6].upper()
        return f"NOM{hash_id}"
    except Exception as e:
        log.warning(f"No se pudo generar ID para el nombre '{nombre}': {e}. Devolviendo None.")
        return None
    
"""
    Cargamos los datos desde un archivo CSV usando nombres de columna definidos manualmente.
    Maneja errores de encoding y existencia de archivo. Asume que la primera fila
    del archivo es un encabezado que debe ser ignorado.


"""
def cargar_datos_csv(ruta: Path, nombres_columnas: list):

    log.info(f"Intentando cargar datos desde: {ruta}")
    if not ruta.is_file():
        log.error(f"La ruta de entrada no es un archivo válido: {ruta}")
        return None

    df = None
    try:
        # Hacemos una caarga forzanda de los nombres e ignorando header original (skiprows=1).
        df = pd.read_csv(
            ruta,
            encoding='utf-8',
            names=nombres_columnas,
            header=None,
            skiprows=1,
            low_memory=False
        )
        log.info(f"Archivo cargado con UTF-8 (encabezado manual, 1 fila saltada).")
    except UnicodeDecodeError:
        log.warning(f"Fallo de encoding UTF-8 en '{ruta.name}'. Intentando Latin-1...")
        try:
            df = pd.read_csv(
                ruta,
                encoding='latin1',
                names=nombres_columnas,
                header=None,
                skiprows=1,
                low_memory=False
            )
            log.warning(f"Archivo cargado con Latin-1 (encabezado manual, 1 fila saltada).")
        except Exception as e_latin:
            log.exception(f"Fallo al cargar con Latin-1: {e_latin}") 
            return None
    except Exception as e_main:
        log.exception(f"Error inesperado durante la carga CSV: {e_main}")
        return None

    if df is None or df.empty:
        log.warning("DataFrame resultante está vacío o es None tras intentos de carga.")
        return df 

    log.info(f"Carga completada: {df.shape[0]} filas, {df.shape[1]} columnas.")
    log.debug(f"Columnas aplicadas manualmente: {df.columns.tolist()}")
    return df

"""
    Aplicamos el pipeline de transformación al DataFrame: limpieza de nombres,
    renombrado, conversión de tipos y correcciones específicas.

"""
def transformar_datos(df: pd.DataFrame):

    if df is None or df.empty:
        log.error("Input inválido para transformar_datos (None o vacío).")
        return pd.DataFrame()

    log.info("Inicio de la fase de transformación...")
    # 1. Limpiar nombres de columna (los aplicados manualmente)
    try:
        df.columns = [limpiar_nombre_columna(col) for col in df.columns]
        log.info(f"Columnas limpiadas a snake_case: {df.columns.tolist()}")
    except Exception as e:
        log.exception(f"Error al limpiar nombres de columna: {e}")
        return pd.DataFrame() # Fallo crítico

    # 2. Renombrar a esquema en español
    cols_antes = df.columns.tolist()
    df.rename(columns=TRADUCCIONES_COLUMNAS, inplace=True)
    cols_despues = df.columns.tolist()
    if cols_antes == cols_despues: log.warning("Renombrado no afectó columnas. Verificar TRADUCCIONES_COLUMNAS.")
    else: log.info(f"Columnas renombradas a español: {cols_despues}")

    # 3. Conversión de tipos
    log.info("Aplicando conversiones de tipo de dato...")
    # Obtener nombres de columnas traducidos para cada tipo
    cols_fecha_es = [v for k, v in TRADUCCIONES_COLUMNAS.items() if k in COLUMNAS_FECHA and v in df.columns]
    cols_num_es = [v for k, v in TRADUCCIONES_COLUMNAS.items() if k in COLUMNAS_NUMERICAS and v in df.columns]
    cols_cat_es_orig = [v for k, v in TRADUCCIONES_COLUMNAS.items() if k in COLUMNAS_CATEGORICAS and v in df.columns]
    cols_cat_es = cols_cat_es_orig + [col for col in COLUMNAS_CATEGORICAS_ADICIONALES if col in df.columns]
    cols_cat_es = list(dict.fromkeys(cols_cat_es))

    for col in cols_fecha_es:
        try:
            df[col] = df[col].replace('01/01/1900', pd.NaT)
            df[col] = pd.to_datetime(df[col], errors='coerce')
        except Exception as e: log.warning(f"Error convirtiendo '{col}' a fecha: {e}")
    for col in cols_num_es:
        try:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        except Exception as e: log.warning(f"Error convirtiendo '{col}' a numérico: {e}")

    # Conversiones específicas
    col_cp = 'codigo_postal'; col_tel = 'telefono'; col_lat = 'latitud'; col_lon = 'longitud'
    try:
        if col_cp in df.columns: df[col_cp] = df[col_cp].astype(str).str.split('.').str[0].str.zfill(5)
        if col_tel in df.columns: df[col_tel] = df[col_tel].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
        if col_lat in df.columns: df[col_lat] = df[col_lat].replace(0.0, np.nan)
        if col_lon in df.columns: df[col_lon] = df[col_lon].replace(0.0, np.nan)
    except Exception as e: log.warning(f"Error en conversiones específicas (CP, Tel, Lat/Lon): {e}")

    # Categóricas (optimizan memoria)
    log.info("Aplicando tipo 'category'...")
    for col in cols_cat_es:
        try:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
                df[col].replace(['nan', 'NaN', 'None', 'Missing', 'Unspecified', '0', ''], np.nan, inplace=True)
                df[col] = df[col].astype('category')
        except Exception as e: log.warning(f"Error convirtiendo '{col}' a categoría: {e}")


    log.info("Fase de transformación completada.")
    return df


def actualizar_dimension_nombres(df: pd.DataFrame, ruta_dim: Path):
    """
    Gestionamos la actualización de la dimensión de nombres: carga, identifica nuevos, genera IDs,
    y retorna la dimensión actualizada.


    """
    log.info("Inicio de actualización de dimensión de nombres...")
    col_nombre = 'nombre_restaurante'
    if col_nombre not in df.columns:
        log.error(f"Columna '{col_nombre}' ausente. No se puede actualizar dimensión.")
        try: return pd.read_csv(ruta_dim, dtype={'id_nombre': str, 'nombre': str})
        except Exception: return pd.DataFrame(columns=['id_nombre', 'nombre'])

    # Carga robusta de la dimensión existente
    dim_nombres = pd.DataFrame(columns=['id_nombre', 'nombre']) # Default vacío
    if ruta_dim.is_file():
        try:
            dim_nombres = pd.read_csv(ruta_dim, dtype={'id_nombre': str, 'nombre': str})
            log.info(f"Dimensión cargada desde '{ruta_dim.name}': {dim_nombres.shape[0]} registros.")
            if not all(c in dim_nombres.columns for c in ['id_nombre', 'nombre']):
                 log.error("Dimensión cargada inválida (faltan columnas). Se reseteará.")
                 dim_nombres = pd.DataFrame(columns=['id_nombre', 'nombre'])
        except Exception as e:
            log.exception(f"Error al cargar dimensión existente '{ruta_dim.name}'. Se usará vacía.")
            dim_nombres = pd.DataFrame(columns=['id_nombre', 'nombre'])
    else:
        log.warning(f"Archivo de dimensión '{ruta_dim.name}' no encontrado. Se creará uno nuevo.")

    # Identificación de nuevos nombres
    nombres_actuales = df[col_nombre].dropna().unique()
    nombres_existentes = set(dim_nombres['nombre'])
    nuevos_nombres = [n for n in nombres_actuales if n not in nombres_existentes]

    if not nuevos_nombres:
        log.info("No se encontraron nuevos nombres para agregar a la dimensión.")
        return dim_nombres # Retornar la cargada sin cambios

    log.info(f"Identificados {len(nuevos_nombres)} nuevos nombres. Generando IDs...")
    nuevos_registros = [{'id_nombre': generar_id_nombre(n), 'nombre': n} for n in nuevos_nombres]
    df_nuevos = pd.DataFrame(nuevos_registros).dropna(subset=['id_nombre']).drop_duplicates(subset=['nombre'])

    # Detección y manejo (log) de colisiones de hash
    id_counts = df_nuevos['id_nombre'].value_counts()
    colisiones = id_counts[id_counts > 1]
    if not colisiones.empty:
        log.warning(f"¡Colisión de HASH detectada para IDs: {colisiones.index.tolist()}!")
        df_nuevos = df_nuevos.drop_duplicates(subset=['id_nombre'], keep='first') # Mitigación simple

    # Concatenamosa y aseguramos la unicidad final por nombre
    dim_actualizada = pd.concat([dim_nombres, df_nuevos], ignore_index=True)
    dim_actualizada = dim_actualizada.drop_duplicates(subset=['nombre'], keep='last')
    log.info(f"Dimensión actualizada a {dim_actualizada.shape[0]} registros.")

    return dim_actualizada

"""
    Esta función aplica el id_nombre de la dimensión al DataFrame principal.


"""

def aplicar_clave_surrogada(df: pd.DataFrame, dim_nombres: pd.DataFrame):

    log.info("Aplicando clave subrogada 'id_nombre'...")
    col_nombre = 'nombre_restaurante'; col_id = 'id_nombre'

    if col_nombre not in df.columns: log.error("Falta 'nombre_restaurante'."); return df
    if dim_nombres.empty or not all(c in dim_nombres.columns for c in ['nombre', col_id]):
        log.error("Dimensión inválida para mapeo."); return df

    # Creamos el mapa de mapeo
    mapa_id = dim_nombres.drop_duplicates(subset=['nombre'], keep='last').set_index('nombre')[col_id]
    df[col_id] = df[col_nombre].map(mapa_id)
    log.info(f"Columna '{col_id}' aplicada/actualizada.")

    # Reportamos las estadísticas de mapeo
    if not df.empty:
        nulos = df[col_id].isnull().sum()
        total = len(df)
        log.info(f"Mapeo ID: {total - nulos}/{total} ({((total - nulos) / total) * 100 if total else 0:.2f}%) OK.")
        if nulos > 0: log.warning(f"{nulos} registros sin ID mapeado.")
    return df
"""
Guardamos los DataFrames finales (principal y dimensión) en formato CSV.

    """
def guardar_salida_csv(df: pd.DataFrame, dim_nombres: pd.DataFrame, ruta_fact_csv: Path, ruta_dim: Path):
   
    log.info("Inicio de la fase de guardado de salidas CSV...")

    # Guardamos el archivo de dimensión
    try:
        dim_nombres.to_csv(ruta_dim, index=False, encoding='utf-8-sig')
        log.info(f"Dimensión actualizada guardada en: {ruta_dim}")
    except Exception as e:
        log.exception(f"Error CRÍTICO al guardar dimensión en '{ruta_dim}': {e}")

    # Guardamos la  tabla principal
    if df is None or df.empty:
        log.warning("DataFrame final vacío. No se guardará archivo principal CSV.")
        return
    try:
        # Reordenamos el orden de las columnas para facilitar la lectura
        columnas_ordenadas = [
            'id_restaurante', 'id_nombre', 'nombre_restaurante', 'fecha_inspeccion',
            'fecha_calificacion', 'fecha_registro', 'calificacion', 'puntuacion', 'accion',
            'es_critica', 'codigo_violacion', 'descripcion_violacion', 'tipo_cocina',
            'tipo_inspeccion', 'distrito', 'edificio', 'calle', 'codigo_postal',
            'telefono', 'latitud', 'longitud', 'punto_ubicacion', 'distrito_comunitario',
            'distrito_consejo', 'zona_censo', 'nta', 'bin', 'bbl'
        ]
        cols_finales = [col for col in columnas_ordenadas if col in df.columns]
        cols_otras = [col for col in df.columns if col not in cols_finales]
        df_guardar = df[cols_finales + cols_otras]

        df_guardar.to_csv(ruta_fact_csv, index=False, encoding='utf-8-sig')
        log.info(f"Datos procesados guardados como CSV en: {ruta_fact_csv}")
    except Exception as e:
        log.exception(f"Error CRÍTICO al guardar datos procesados CSV en '{ruta_fact_csv}': {e}")

# --- Flujo Principal de Ejecución ---
def main():
    """Orquesta el flujo completo del ETL."""
    log.info("=========================================")
    log.info("=== INICIO ETL: Inspecciones NYC ===")
    log.info("=========================================")

    # 0. Validación Inicial de Archivo de Entrada
    log.info(f"Verificando archivo de entrada: {RUTA_CSV_INSPECCIONES}")
    if not RUTA_CSV_INSPECCIONES.is_file():
        log.critical(f"¡ABORTANDO! Archivo de entrada no encontrado en: {RUTA_CSV_INSPECCIONES}")
        sys.exit(1) # Salir con código de error
    log.info("Archivo de entrada encontrado.")

    # 1. Extracción
    log.info("--- Fase 1: Extracción ---")
    # Nombres de columnas definidos manualmente para la carga
    nombres_columnas_originales = [
        "CAMIS", "DBA", "BORO", "BUILDING", "STREET", "ZIPCODE", "PHONE", "CUISINE DESCRIPTION",
        "INSPECTION DATE", "ACTION", "VIOLATION CODE", "VIOLATION DESCRIPTION", "CRITICAL FLAG",
        "SCORE", "GRADE", "GRADE DATE", "RECORD DATE", "INSPECTION TYPE", "Latitude", "Longitude",
        "Community Board", "Council District", "Census Tract", "BIN", "BBL", "NTA", "Location Point1"
    ]
    df_raw = cargar_datos_csv(RUTA_CSV_INSPECCIONES, nombres_columnas_originales)

    if df_raw is None or df_raw.empty:
        log.critical("Fallo en la extracción o archivo vacío. Abortando ETL.")
        sys.exit(1)

    # 2. Transformación
    log.info("--- Fase 2: Transformación ---")
    df_transformed = transformar_datos(df_raw.copy()) # Usar copia para preservar raw

    if df_transformed.empty:
        log.critical("Transformación resultó en DataFrame vacío. Abortando ETL.")
        sys.exit(1)

    # 3. Actualización de Dimensión
    log.info("--- Fase 3: Actualización Dimensión Nombres ---")
    dim_nombres = actualizar_dimension_nombres(df_transformed, RUTA_DIM_NOMBRES)

    # 4. Aplicación de Clave Subrogada
    log.info("--- Fase 4: Aplicación Clave Subrogada ---")
    df_final = aplicar_clave_surrogada(df_transformed, dim_nombres)

    # 5. Carga (Guardado)
    log.info("--- Fase 5: Carga (Guardado de Salidas) ---")
    guardar_salida_csv(df_final, dim_nombres, RUTA_SALIDA_PROCESADO_CSV, RUTA_SALIDA_DIM_ACTUALIZADA)

    log.info("===========================================")
    log.info("=== ETL completado satisfactoriamente ===")
    log.info("===========================================")

    # Opcional: Mostrar muestra final para verificación visual rápida
    if 'nombre_restaurante' in df_final.columns and 'id_nombre' in df_final.columns:
        log.info("Muestra de datos finales (Nombre vs ID):")
        print(df_final[df_final['id_nombre'].notna()][['nombre_restaurante', 'id_nombre']].head())

if __name__ == "__main__":
    # Este bloque asegura que main() solo se ejecute cuando corres el script directamente.
    main()