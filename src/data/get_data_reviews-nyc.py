import os
import glob
import json
import pandas as pd
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from path_manager import get_data_paths

def aplanar_json(datos, prefijo=""):
    resultado = {}
    for clave, valor in datos.items():
        nueva_clave = f"{prefijo}{clave}"
        if isinstance(valor, dict):
            resultado.update(aplanar_json(valor, f"{nueva_clave}_"))
        elif isinstance(valor, list):
            for i, item in enumerate(valor):
                if isinstance(item, (dict, list)):
                    resultado.update(aplanar_json({str(i): item}, f"{nueva_clave}_"))
                else:
                    resultado[f"{nueva_clave}_{i}"] = item
        else:
            resultado[nueva_clave] = valor
    return resultado

def procesar_reviews(carpeta_json, salida_dir, nombre_salida='reviews.parquet'):
    """
    Procesa archivos JSON dentro de carpeta_json y guarda un archivo Parquet en salida_dir.
    """
    os.makedirs(salida_dir, exist_ok=True)
    archivos_json = glob.glob(os.path.join(carpeta_json, '**', '*.json'), recursive=True)
    print(f"📁 Se encontraron {len(archivos_json)} archivos JSON en todas las subcarpetas.")

    registros = []

    for ruta_json in archivos_json:
        estado = os.path.basename(os.path.dirname(ruta_json)).replace('review-', '')
        print(f"📄 Procesando archivo: {ruta_json} (Estado: {estado})")

        try:
            with open(ruta_json, 'r', encoding='utf-8') as f:
                for linea in f:
                    try:
                        data = json.loads(linea)
                        data_flat = aplanar_json(data)
                        data_flat['estado'] = estado
                        registros.append(data_flat)
                    except Exception as e:
                        print(f"⚠️ Error leyendo línea en {ruta_json}: {e}")
        except Exception as e:
            print(f"❌ Error abriendo archivo {ruta_json}: {e}")

    if registros:
        df_final = pd.DataFrame(registros)
        ruta_parquet = os.path.join(salida_dir, nombre_salida)
        df_final.to_parquet(ruta_parquet, index=False)
        print(f"✅ Archivo generado: {ruta_parquet} con {len(df_final)} registros.")
        return ruta_parquet
    else:
        print("⚠️ No se encontraron registros válidos para guardar.")
        return None

# ========= EJECUCIÓN DIRECTA =========
if __name__ == "__main__":
    paths = get_data_paths()
    carpeta_json = os.path.join(paths['external'], 'review-New_York')
    salida_dir = paths['interim']
    procesar_reviews(carpeta_json, salida_dir, 'review_New_York.parquet')
