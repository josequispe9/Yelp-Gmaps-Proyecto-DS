import pandas as pd
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from path_manager import get_data_paths
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import numpy as np


paths = get_data_paths()  

path_raw = paths['raw']
path_processed = paths['processed']
path_interim = paths['interim']
path_external = paths['external']

# path para cargar el dataset
path_dataset = os.path.join(path_interim, 'metadata_sitios.parquet')
path_coord_nyc = os.path.join(path_raw, 'coord_nyc/nybb.shp')

# path para guardar el dataset limpio
path_dataset_clean = os.path.join(path_interim, 'restaurantes_nyc_google.parquet')

palabras_clave_restaurantes = [
    "restaurant", "cafe", "bar", "bakery", "grill", "food", "diner", "pizza",
    "coffee", "tea", "sushi", "seafood", "bistro", "steak", "chicken", "fast food",
    "sandwich", "pub", "ice cream", "taqueria", "pizzeria", "brunch"
]


def filtro_geografico(df, coord_nyc):
    # Filtrar por NYC sin modificar el DataFrame original
    geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]
    df_geo = gpd.GeoDataFrame(df.copy(), geometry=geometry, crs="EPSG:4326")  # Crear un GeoDataFrame temporal
    df_geo = df_geo.to_crs(coord_nyc.crs)  # Convertir al CRS de NYC
    
    # Realizar la intersección geoespacial
    df_filtered = gpd.sjoin(df_geo, coord_nyc, how='inner', predicate='within')  # Filtrar por la región de NYC

    # Retornar el DataFrame filtrado sin columnas adicionales (como 'geometry')
    # Filtrar solo las columnas necesarias para que coincidan con las del DataFrame original
    columnas_originales = df.columns
    return df_filtered[columnas_originales]  # Solo las columnas originales del df

def filtro_categoria(df):
    def contiene_categoria_relacionada(categorias):
        if not isinstance(categorias, (list, np.ndarray)):
            return False
        for cat in categorias:
            cat_lower = cat.lower()
            if any(palabra in cat_lower for palabra in palabras_clave_restaurantes):
                return True
        return False

    filtro = df['category'].apply(contiene_categoria_relacionada)
    return df[filtro]



def limpiar_dataset_negocios(path_dataset, path_coord_nyc):
    # Cargar datos
    df = pd.read_parquet(path_dataset)
    coord_nyc = gpd.read_file(path_coord_nyc)
    
    
    # Eliminar duplicados y columnas no necesarias
    df = df.drop_duplicates(subset=['gmap_id'], keep='first')
    df = df.drop(columns=['relative_results'], errors='ignore')  

    # Filtrar solo negocios relacionados con comida
    df_filtrado = filtro_categoria(df)    

    # Filtrar por coordenadas de NYC
    df_restaurantes = filtro_geografico(df_filtrado, coord_nyc)
    
    return df_restaurantes

def visualizar(df_restaurantes, coord_nyc):
    fig, ax = plt.subplots(figsize=(10, 10))
    coord_nyc.plot(ax=ax, edgecolor='black', facecolor='lightgray')
    df_restaurantes.plot(ax=ax, color='red', markersize=2)
    ax.set_title("Negocios dentro de NYC", fontsize=15)
    ax.set_axis_off()
    plt.show()

    return None


# Cargar el dataset limpio
df_restaurantes = limpiar_dataset_negocios(path_dataset, path_coord_nyc)

# Guardar el dataset limpio sin agregar nuevas columnas
df_restaurantes.to_parquet(path_dataset_clean, index=False)
