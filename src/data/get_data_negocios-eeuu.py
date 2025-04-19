import os
import glob
import json
import pandas as pd
import sys

# Agregar el directorio raíz al path para poder importar path_manager
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from path_manager import get_data_paths

def main():
    paths = get_data_paths()  # Ahora sí se define correctamente

    # ========== RUTAS ==========
    carpeta_json = os.path.join(paths['external'], 'metadata-sitios')
    salida_dir = paths['interim']
    os.makedirs(salida_dir, exist_ok=True)

    # ========== CONFIG ==========
    CAMPOS_OBJETIVO = [
        'name', 'address', 'gmap_id', 'description',
        'latitude', 'longitude', 'category', 'avg_rating', 'num_of_reviews', 'price',
        'hours', 'MISC', 'state', 'relative_results', 'url'
    ]

    # Función interna: extrae solo los campos necesarios
    def filtrar_campos(json_obj):
        return {campo: json_obj.get(campo, None) for campo in CAMPOS_OBJETIVO}

    # Leer archivos JSON y filtrar datos
    archivos_json = glob.glob(os.path.join(carpeta_json, '*.json'))
    print(f"📁 Se encontraron {len(archivos_json)} archivos JSON.")

    registros = []

    for ruta_json in archivos_json:
        print(f"📄 Procesando archivo: {os.path.basename(ruta_json)}")
        with open(ruta_json, 'r', encoding='utf-8') as f:
            for linea in f:
                try:
                    data = json.loads(linea)
                    registros.append(filtrar_campos(data))
                except Exception as e:
                    print(f"⚠️ Error leyendo línea en {ruta_json}: {e}")

    # Crear DataFrame y exportar a Parquet
    df_final = pd.DataFrame(registros)
    ruta_parquet = os.path.join(salida_dir, "metadata_sitios.parquet")
    df_final.to_parquet(ruta_parquet, index=False)
    print(f"✅ Parquet generado: {ruta_parquet} con {len(df_final)} registros.")

if __name__ == "__main__":
    main()
