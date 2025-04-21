import pandas as pd
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from path_manager import get_data_paths
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import numpy as np
import ast

paths = get_data_paths()  

path_raw = paths['raw']
path_processed = paths['processed']
path_interim = paths['interim']
path_external = paths['external']

# path para cargar el dataset
path_dataset = os.path.join(path_interim, 'metadata_sitios.parquet')
path_coord_nyc = os.path.join(path_raw, 'coord_nyc/nybb.shp')
# path para guardar el dataset limpio
path_dataset_clean = os.path.join(path_processed, 'limpieza_negocios_eeuu.csv')

palabras_clave_restaurantes = [
    "restaurant", "cafe", "bar", "bakery", "grill", "food", "diner", "pizza",
    "coffee", "tea", "sushi", "seafood", "bistro", "steak", "chicken", "fast food",
    "sandwich", "pub", "ice cream", "taqueria", "pizzeria", "brunch"
]


def filtrar_dataset(df, coord_nyc):
    
    # Filtrar por   NYC
    geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]
    df_geo = gpd.GeoDataFrame(df.copy(), geometry=geometry, crs="EPSG:4326")  # Coordenadas geográficas
    df_geo = df_geo.to_crs(coord_nyc.crs)
    df_filtered = gpd.sjoin(df_geo, coord_nyc, how='inner', predicate='within')
    
    df_filtered

    return df_filtered


def filtrar_restaurantes(df):
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
    df = df.drop(columns=['gmap_id', 'relative_results'], errors='ignore')  # Por si alguna no está

    
    # Filtrar por coordenadas de NYC
    df_filtrado = filtrar_dataset(df, coord_nyc)
    
    # Filtrar solo negocios relacionados con comida
    df_restaurantes = filtrar_restaurantes(df_filtrado)

    return df_restaurantes

def visualizar():
    fig, ax = plt.subplots(figsize=(10, 10))
    coord_nyc.plot(ax=ax, edgecolor='black', facecolor='lightgray')
    df_restaurantes.plot(ax=ax, color='red', markersize=2)
    ax.set_title("Negocios dentro de NYC", fontsize=15)
    ax.set_axis_off()
    plt.show()
    
    return None

limpiar_dataset_negocios(path_dataset, path_coord_nyc)

df_restaurantes.info()








import seaborn as sns
from collections import Counter
import contextily as ctx

def get_report(df):
    sns.set(style="whitegrid")

    # -------------------- 1. Histograma de Ratings --------------------
    plt.figure(figsize=(8, 4))
    sns.histplot(df['avg_rating'], bins=20, kde=True)
    plt.title("Distribución de Ratings")
    plt.xlabel("Rating Promedio")
    plt.ylabel("Cantidad de Restaurantes")
    plt.tight_layout()
    plt.show()

    # -------------------- 2. Histograma de Número de Reseñas --------------------
    plt.figure(figsize=(8, 4))
    sns.histplot(df['num_of_reviews'], bins=30, log_scale=(False, True))
    plt.title("Cantidad de Reseñas (escala log)")
    plt.xlabel("Número de Reseñas")
    plt.ylabel("Frecuencia (log)")
    plt.tight_layout()
    plt.show()

    # -------------------- 3. Distribución de Rangos de Precio --------------------
    if 'price' in df.columns:
        plt.figure(figsize=(8, 4))
        sns.countplot(data=df, x='price', order=df['price'].value_counts().index)
        plt.title("Distribución de Rangos de Precio")
        plt.xlabel("Rango de Precio")
        plt.ylabel("Cantidad de Restaurantes")
        plt.tight_layout()
        plt.show()

    # -------------------- 4. Top 20 Categorías --------------------
    if 'category' in df.columns:
        try:
            categorias_planas = np.concatenate(df['category'].dropna().values)
            conteo_categorias = Counter(categorias_planas)
            top_categorias = conteo_categorias.most_common(20)
            categorias, cantidades = zip(*top_categorias)

            plt.figure(figsize=(10, 6))
            sns.barplot(x=cantidades, y=categorias)
            plt.title("Top 20 Categorías de Restaurantes")
            plt.xlabel("Frecuencia")
            plt.ylabel("Categoría")
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print("Error al procesar categorías:", e)

    # -------------------- 5. Mapa de Calor de Ratings --------------------
    try:
        gdf_webmerc = df.to_crs(epsg=3857)

        fig, ax = plt.subplots(figsize=(10, 10))
        gdf_webmerc.plot(ax=ax, column='avg_rating', cmap='coolwarm', legend=True,
                         markersize=5, alpha=0.7)
        ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
        plt.title("Mapa de Calor de Ratings en NYC")
        plt.axis("off")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print("No se pudo generar el mapa:", e)

get_report(df_restaurantes)