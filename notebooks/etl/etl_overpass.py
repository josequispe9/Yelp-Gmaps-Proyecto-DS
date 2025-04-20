
# Librerias
import pandas as pd
import numpy as np
import hashlib # Necesario para generar IDs de nombre
import re
import unicodedata
import logging
from pathlib import Path
import sys
from io import StringIO # Para capturar salida de df.info()
import json # <-- Importante para leer JSON

# --- Configuración de Logging  ---
# Configuracion de logger para seguir el flujo del script y diagnosticar problemas.
logging.basicConfig(
    level=logging.INFO, # Nivel mínimo: INFO
    format='%(asctime)s [%(levelname)s] %(name)s.%(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger(__name__) # Logger específico

# --- Definición de Rutas y Constantes ---
# Definimos las rutas de forma centralizada y absoluta usando pathlib.
try:

    RUTA_BASE = Path(r"H:\git\proyecto grupal 2\Yelp-Gmaps-Proyecto-DS").resolve()
    RUTA_RAW = RUTA_BASE / "data" / "raw"
    RUTA_PROCESSED = RUTA_BASE / "data" / "processed"
    RUTA_DIM = RUTA_PROCESSED / "dim"

    # Cargamos la ruta del JSON de OSM desde Overpass API
    RUTA_JSON_OSM = (RUTA_RAW / "overpass-turbo.eu" / "export.json").resolve()

    # Cargamos la ruta del CSV de dimensión de nombres
    RUTA_DIM_NOMBRES = (RUTA_DIM / "dim_nombre_restaurante_limpia.csv").resolve()

    # Ruta de salida para los datos OSM procesados en CSV
    RUTA_SALIDA_OSM_CSV = (RUTA_PROCESSED / "osm_restaurantes_processed.csv").resolve()

    # Creaamos directorios necesarios
    RUTA_PROCESSED.mkdir(parents=True, exist_ok=True)
    RUTA_DIM.mkdir(parents=True, exist_ok=True)

except Exception as e:
    log.exception(f"Error crítico al definir o resolver rutas: {e}")
    sys.exit(1)

# Creamos un dicionario com los nombres de las columnas y su traduccion
TRADUCCIONES_COLUMNAS_OSM = {
    'osm_id': 'id_osm',
    'osm_type': 'tipo_osm',
    'lat': 'latitud',
    'lon': 'longitud',
    'tag_name': 'nombre_establecimiento',
    'tag_amenity': 'tipo_lugar_osm',
    'tag_cuisine': 'descripcion_cocina', 
    'tag_addr_housenumber': 'numero_calle',
    'tag_addr_street': 'calle',
    'tag_addr_city': 'ciudad',
    'tag_addr_postcode': 'codigo_postal',
    'tag_addr_state': 'estado',
    'tag_phone': 'telefono',
    'tag_website': 'website',
    'tag_opening_hours': 'horario_osm',
    'tag_brand': 'marca_osm',
    'tag_wheelchair': 'accesibilidad_silla_ruedas_osm',
    'tag_takeaway': 'para_llevar_osm',
    'tag_delivery': 'entrega_domicilio_osm',
}

# --- Funciones Auxiliares ---
"""Estandarizamos los nombres de las columnas a formato snake_case."""
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

"""Generamos un ID hash corto determinista tipo NOMXXXXXX para nuevos nombres."""
def generar_id_nombre(nombre: str) -> str | None:
    
    if pd.isna(nombre) or not str(nombre).strip(): return None
    try:
        nombre_std = str(nombre).strip().upper()
        hash_id = hashlib.md5(nombre_std.encode('utf-8')).hexdigest()[:6].upper()
        return f"NOM{hash_id}"
    except Exception as e: log.warning(f"No se pudo generar ID para '{nombre}': {e}."); return None

# --- Funciones de Procesamiento para OSM JSON ---
"""
    Cargamos datos JSON de Overpass API, filtramos los elementos relevantes (nodos
    de restaurantes/cafes/etc.) y aplanamos la estructura extrayendo tags.

"""
def cargar_y_aplanar_osm_json(ruta_json: Path) -> pd.DataFrame | None:

    log.info(f"Intentando cargar y aplanar JSON desde: {ruta_json}")
    if not ruta_json.is_file(): log.error(f"Ruta JSON no es archivo: {ruta_json}"); return None

    try:
        with open(ruta_json, 'r', encoding='utf-8') as f: data = json.load(f)
    except Exception as e: log.exception(f"Error leyendo/decodificando JSON: {e}"); return None

    elementos = data.get('elements', [])
    if not elementos: log.warning("JSON sin clave 'elements' o vacía."); return pd.DataFrame()

    log.info(f"Procesando {len(elementos)} elementos JSON...")
    datos_aplanados = []
    amenities_interes = {'restaurant', 'cafe', 'fast_food', 'pub', 'bar', 'food_court', 'ice_cream'}
    # Lista de tags OSM que queremos extraer (sin el prefijo 'tag_')
    tags_a_extraer = [
        'name', 'amenity', 'cuisine', 'addr:housenumber', 'addr:street',
        'addr:city', 'addr:postcode', 'addr:state', 'phone', 'website',
        'opening_hours', 'brand', 'wheelchair', 'takeaway', 'delivery'
    ]

    for elem in elementos:
        if elem.get('type') == 'node' and 'tags' in elem:
            tags = elem.get('tags', {})
            amenity = tags.get('amenity')
            if amenity in amenities_interes:
                registro = {
                    'osm_id': elem.get('id'),
                    'osm_type': elem.get('type'),
                    'lat': elem.get('lat'),
                    'lon': elem.get('lon'),
                }
                # Extraer tags de interés de forma dinámica
                for tag_key in tags_a_extraer:
                    # Creamos el nombre de columna prefijado 'tag_' para el dict intermedio
                    col_key = f"tag_{tag_key.replace(':', '_')}" # tag_addr_housenumber
                    registro[col_key] = tags.get(tag_key) # Usa .get() para manejar ausencias

                datos_aplanados.append(registro)

    if not datos_aplanados: log.warning("No se extrajeron nodos relevantes."); return pd.DataFrame()

    log.info(f"Se extrajeron {len(datos_aplanados)} registros relevantes.")
    df_aplanado = pd.DataFrame(datos_aplanados)
    return df_aplanado
"""Pipeline de transformación para datos OSM aplanados."""
def transformar_datos_osm(df: pd.DataFrame) -> pd.DataFrame:

    if df is None or df.empty: log.error("Input inválido."); return pd.DataFrame()
    log.info("Inicio transformación datos OSM...")

    # 1. Limpiar nombres de columna (ej. 'tag_addr_housenumber')
    try: df.columns = [limpiar_nombre_columna(col) for col in df.columns]
    except Exception as e: log.exception("Error limpieza nombres OSM."); return pd.DataFrame()
    log.info(f"Columnas OSM limpiadas: {df.columns.tolist()}")

    # 2. Renombramos a español estándar
    try:
        nuevos_nombres = [TRADUCCIONES_COLUMNAS_OSM.get(col, col) for col in df.columns]
        renombradas_count = sum(1 for old, new in zip(df.columns, nuevos_nombres) if old != new)
        df.columns = nuevos_nombres
        if renombradas_count == 0: log.warning("Ninguna col OSM renombrada.")
        else: log.info(f"{renombradas_count} cols OSM renombradas a estándar.")
        log.info(f"Nombres finales OSM (estándar): {df.columns.tolist()}")
    except Exception as e: log.exception("Error renombrado manual OSM."); return pd.DataFrame()

    # 3. Conversión de tipos de datos y limpieza usando  losnombres estándar
    log.info("Aplicando conversiones de tipo y limpieza OSM...")
    if 'id_osm' in df.columns: df['id_osm'] = df['id_osm'].astype(str)
    if 'tipo_osm' in df.columns: df['tipo_osm'] = df['tipo_osm'].astype('category')
    for col in ['latitud', 'longitud']:
        if col in df.columns:
             try: df[col] = pd.to_numeric(df[col], errors='coerce')
             except Exception as e: log.warning(f"Error conversión {col}: {e}")

    # Categorías usando los nombres estandar
   
    for col in ['tipo_lugar_osm', 'descripcion_cocina', 'ciudad', 'estado', 'marca_osm']:
        if col in df.columns:
            try: df[col] = df[col].astype(str).str.strip().replace(['nan','None',''], np.nan).astype('category')
            except Exception as e: log.warning(f"Error conversión {col} a category: {e}")
    # Strings
    cols_string = ['nombre_establecimiento', 'numero_calle', 'calle', 'codigo_postal',
                   'telefono', 'website', 'horario_osm']
    for col in cols_string:
        if col in df.columns:
            try:
                df[col] = df[col].astype(str).str.strip().replace(['nan','None'], '')
                df[col] = df[col].astype('string')
            except Exception as e: log.warning(f"Error limpieza/conversión {col} a string: {e}")
    # Booleanos
    mapa_si_no_osm = {'yes': True, 'no': False, 'limited': None, 'true': True, 'false': False}
    for col in ['accesibilidad_silla_ruedas_osm', 'para_llevar_osm', 'entrega_domicilio_osm']:
        if col in df.columns:
             try: df[col] = df[col].astype(str).str.lower().str.strip().map(mapa_si_no_osm).astype('boolean')
             except Exception as e: log.warning(f"Error conversión {col} a booleano: {e}")

    # Eliminamos filas sin nombre estándar válido
    col_nombre_std = 'nombre_establecimiento'
    if col_nombre_std in df.columns:
        original_rows = len(df)
        df = df[df[col_nombre_std].astype(str).str.strip() != '']
        df.dropna(subset=[col_nombre_std], inplace=True)
        rows_dropped = original_rows - len(df)
        if rows_dropped > 0: log.info(f"Eliminadas {rows_dropped} filas sin nombre válido en '{col_nombre_std}'.")

    log.info("Transformación OSM completada.")
    return df

# --- Funciones de Dimensión y Guardado (Reutilizadas) ---

def gestionar_dimension_nombres_con_flag(
    df_fuente: pd.DataFrame, ruta_dim: Path, col_nombre_fuente: str,
    flag_permiso_temporal: bool = False
) -> pd.DataFrame:
    log.info(f"Inicio gestión dimensión nombres desde '{col_nombre_fuente}'. Flag Temporal={flag_permiso_temporal}.")
    col_flag = 'permiso_temporal'
    dim_nombres = pd.DataFrame(columns=['id_nombre', 'nombre', col_flag])
    if ruta_dim.is_file():
        try:
            dim_nombres = pd.read_csv(ruta_dim, dtype={'id_nombre': str, 'nombre': str}, keep_default_na=False)
            log.info(f"Dimensión cargada '{ruta_dim.name}': {dim_nombres.shape[0]} regs.")
            if col_flag not in dim_nombres.columns:
                log.warning(f"'{col_flag}' no encontrada. Añadiendo con default=False.")
                dim_nombres[col_flag] = False
            map_bool = {'true': True, 'false': False, '1': True, '0': False, '1.0': True, '0.0': False, True: True, False: False, np.nan: False, None: False, '':False}
            dim_nombres[col_flag] = dim_nombres[col_flag].astype(str).str.lower().map(map_bool).fillna(False).astype('boolean')
            if not all(c in dim_nombres.columns for c in ['id_nombre', 'nombre', col_flag]):
                 log.error("Dimensión inválida. Reseteando."); dim_nombres = pd.DataFrame(columns=['id_nombre', 'nombre', col_flag])
        except Exception as e: log.exception(f"Error cargando/inicializando dimensión."); dim_nombres = pd.DataFrame(columns=['id_nombre', 'nombre', col_flag])
    else: log.warning(f"Dimensión '{ruta_dim.name}' no encontrada. Creando nueva.")

    if col_flag not in dim_nombres.columns: dim_nombres[col_flag] = False
    dim_nombres[col_flag] = dim_nombres[col_flag].fillna(False).astype('boolean')

    if col_nombre_fuente not in df_fuente.columns: log.error(f"Columna '{col_nombre_fuente}' ausente."); return dim_nombres
    nombres_fuente_unicos = set(n for n in df_fuente[col_nombre_fuente].dropna().unique() if str(n).strip())
    log.info(f"{len(nombres_fuente_unicos)} nombres únicos válidos en fuente.")
    if not nombres_fuente_unicos: log.warning("No hay nombres válidos en fuente."); return dim_nombres

    nombres_existentes_en_dim = set(dim_nombres['nombre'])
    nombres_a_actualizar = nombres_fuente_unicos.intersection(nombres_existentes_en_dim)
    nombres_nuevos = nombres_fuente_unicos - nombres_existentes_en_dim

    if flag_permiso_temporal and nombres_a_actualizar:
        log.info(f"Actualizando '{col_flag}'=True para {len(nombres_a_actualizar)} nombres existentes.")
        mascara_actualizar = dim_nombres['nombre'].isin(nombres_a_actualizar)
        dim_nombres.loc[mascara_actualizar, col_flag] = True
    elif nombres_a_actualizar: log.info(f"{len(nombres_a_actualizar)} nombres ya existen. Flag '{col_flag}' no se modificará.")

    if not nombres_nuevos: log.info("No hay nuevos nombres para agregar."); return dim_nombres
    log.info(f"Identificados {len(nombres_nuevos)} nuevos nombres. Generando IDs...")
    nuevos_registros = [{'id_nombre': gid, 'nombre': n, col_flag: flag_permiso_temporal} for n in nombres_nuevos if (gid := generar_id_nombre(n))]
    if not nuevos_registros: log.warning("No se generaron registros nuevos válidos."); return dim_nombres

    df_nuevos = pd.DataFrame(nuevos_registros).drop_duplicates(subset=['nombre'])
    id_counts = df_nuevos['id_nombre'].value_counts(); colisiones = id_counts[id_counts > 1]
    if not colisiones.empty: log.warning(f"¡Colisión HASH IDs nuevos: {colisiones.index.tolist()}!")
    df_nuevos = df_nuevos.drop_duplicates(subset=['id_nombre'], keep='first')
    df_nuevos[col_flag] = df_nuevos[col_flag].astype('boolean')

    dim_actualizada = pd.concat([dim_nombres, df_nuevos], ignore_index=True)
    dim_actualizada = dim_actualizada.drop_duplicates(subset=['nombre'], keep='last')
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
"""Guardamos el DataFrame de dimensión como CSV."""

def guardar_dimension(dim_df: pd.DataFrame, ruta_dim: Path):
    log.info(f"Inicio guardado dimensión en: {ruta_dim}")
    if 'permiso_temporal' not in dim_df.columns: dim_df['permiso_temporal'] = False
    dim_df['permiso_temporal'] = dim_df['permiso_temporal'].astype('boolean')
    if dim_df is None or dim_df.empty: log.warning("Dimensión vacía, no se guarda."); return
    try:
        ruta_dim.parent.mkdir(parents=True, exist_ok=True)
        dim_cols_ordered = ['id_nombre', 'nombre', 'permiso_temporal']
        dim_df_save = dim_df[[col for col in dim_cols_ordered if col in dim_df.columns]]
        dim_df_save.to_csv(ruta_dim, index=False, encoding='utf-8-sig')
        log.info(f"Dimensión guardada en: {ruta_dim}")
    except Exception as e: log.exception(f"Error CRÍTICO guardando dimensión: {e}")

"""Guardamos el DataFrame procesado final como CSV."""
def guardar_salida_csv(df: pd.DataFrame, ruta_salida: Path):
    log.info(f"Inicio guardado salida CSV en: {ruta_salida}")
    if df is None or df.empty: log.warning("DataFrame vacío, no se guarda."); return
    try:
        ruta_salida.parent.mkdir(parents=True, exist_ok=True)
        # Definir orden preferido usando nombres estándar actualizados
        columnas_ordenadas = [
            'id_osm', 'id_nombre', 'nombre_establecimiento',
            'tipo_lugar_osm', 'descripcion_cocina', # <-- Nombre estándar actualizado
            'latitud', 'longitud',
            'calle', 'numero_calle', 'ciudad', 'distrito', # distrito puede no existir en OSM
            'codigo_postal', 'estado',
            'telefono', 'website',
            'horario_osm', 'marca_osm', 'tipo_osm',
            'accesibilidad_silla_ruedas_osm', 'para_llevar_osm', 'entrega_domicilio_osm'
        ]
        cols_finales = [col for col in columnas_ordenadas if col in df.columns]
        cols_otras = [col for col in df.columns if col not in cols_finales]
        df_guardar = df[cols_finales + cols_otras]
        df_guardar.to_csv(ruta_salida, index=False, encoding='utf-8-sig')
        log.info(f"Datos procesados guardados como CSV en: {ruta_salida}")
    except Exception as e: log.exception(f"Error CRÍTICO guardando CSV en '{ruta_salida}': {e}")


# --- Flujo Principal de Ejecución para OSM ---
def main_osm():
    """Orquesta el flujo ETL para datos OSM JSON."""
    log.info("================================================")
    log.info("=== INICIO ETL: Restaurantes desde OSM JSON (Nombres Estándar) ===")
    log.info("================================================")

    # 0. Validación de entrada
    log.info(f"Verificando entrada OSM JSON: {RUTA_JSON_OSM}")
    if not RUTA_JSON_OSM.is_file(): log.critical(f"Abortando: Archivo OSM JSON no encontrado: {RUTA_JSON_OSM}"); sys.exit(1)
    log.info("Entrada OSM JSON encontrada.")

    # 1. Extracción y Aplanado
    log.info("--- Fase 1: Extracción y Aplanado OSM JSON ---")
    df_raw_osm = cargar_y_aplanar_osm_json(RUTA_JSON_OSM)
    if df_raw_osm is None or df_raw_osm.empty: log.critical("Fallo extracción/aplanado OSM."); sys.exit(1)

    # 2. Transformación
    log.info("--- Fase 2: Transformación Datos OSM ---")
    df_transformed_osm = transformar_datos_osm(df_raw_osm.copy())
    if df_transformed_osm.empty: log.critical("Transformación OSM vacía."); sys.exit(1)

    # 3. Gestión de Dimensión Nombres (Flag Temporal = False)
    log.info("--- Fase 3: Gestión Dimensión Nombres ---")
    columna_nombre_estandar = 'nombre_establecimiento' # <-- Usar nombre estándar
    if columna_nombre_estandar not in df_transformed_osm.columns:
        log.critical(f"Abortando: Falta '{columna_nombre_estandar}' para gestionar dimensión.")
        sys.exit(1)
    dim_nombres_actualizada = gestionar_dimension_nombres_con_flag(
        df_fuente=df_transformed_osm,
        ruta_dim=RUTA_DIM_NOMBRES,
        col_nombre_fuente=columna_nombre_estandar,
        flag_permiso_temporal=False # <-- OSM no implica permiso temporal
    )

    # 4. Guardamos Dimensión Actualizada
    log.info("--- Fase 4: Guardado Dimensión Nombres ---")
    guardar_dimension(dim_nombres_actualizada, RUTA_DIM_NOMBRES) # Sobreescribir

    # 5. Aplicación de Clave Subrogada
    log.info("--- Fase 5: Aplicación Clave Subrogada ---")
    df_final_osm = aplicar_clave_surrogada(
        df=df_transformed_osm,
        dim_nombres=dim_nombres_actualizada,
        col_nombre_origen=columna_nombre_estandar
        )

    # 6. Cargamos 
    log.info("--- Fase 6: Carga (Guardado Salida OSM) ---")
    guardar_salida_csv(df_final_osm, RUTA_SALIDA_OSM_CSV)

    log.info("===================================================")
    log.info("=== ETL Restaurantes OSM (Nombres Estándar) completado ===")
    log.info("===================================================")

    log.info("Info del DataFrame OSM procesado final:")
    buffer = StringIO(); df_final_osm.info(buf=buffer); log.info(buffer.getvalue())
    log.info("Info de la Dimensión de Nombres actualizada final:")
    buffer_dim = StringIO(); dim_nombres_actualizada.info(buf=buffer_dim); log.info(buffer_dim.getvalue())
    log.info(f"Primeras filas de la dimensión actualizada:\n{dim_nombres_actualizada.head().to_string()}")

if __name__ == "__main__":
    main_osm()