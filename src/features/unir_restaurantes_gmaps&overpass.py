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


# Cargar dataframe de overpass
overpass_df = pd.read_parquet(os.path.join(path_interim, 'restaurantes_nyc_overpass.parquet'))
# Cargar dataframe de google maps
gmaps_df = pd.read_parquet(os.path.join(path_interim, 'restaurantes_nyc_google.parquet'))

overpass_df = overpass_df.drop(columns=[ 'gmap_id', 'type', 'price'], errors='ignore')
overpass_df.info()
gmaps_df.info()



overpass_df['type']
gmaps_df['latitude']

overpass_df['gmap_id']
gmaps_df['gmap_id']



from sklearn.neighbors import BallTree
import numpy as np
import pandas as pd

# 1. Convertir coordenadas a radianes
coords_overpass = np.radians(overpass_df[['latitude', 'longitude']])
coords_gmaps = np.radians(gmaps_df[['latitude', 'longitude']])

# 2. Árbol para búsqueda geográfica
tree = BallTree(coords_gmaps, metric='haversine')
dist, idx = tree.query(coords_overpass, k=1)

# 3. Calcular distancia en metros
dist_meters = dist[:, 0] * 6371000
tolerance = 5
matched_mask = dist_meters < tolerance

# 4. Matcheos por coordenadas
overpass_matched = overpass_df[matched_mask].reset_index(drop=True)
gmaps_matched = gmaps_df.iloc[idx[matched_mask][:, 0]].reset_index(drop=True)

# 5. Crear registros combinados (MATCHED)
matched_df = pd.DataFrame()
matched_df['name'] = gmaps_matched['name']
matched_df['address'] = gmaps_matched['address'].combine_first(overpass_matched['address'])
matched_df['description_gmaps'] = gmaps_matched['description']
matched_df['description_overpass'] = overpass_matched['description']
matched_df['latitude'] = gmaps_matched['latitude']
matched_df['longitude'] = gmaps_matched['longitude']
matched_df['category'] = gmaps_matched['category'].combine_first(overpass_matched['category'])
matched_df['hours'] = gmaps_matched['hours'].combine_first(overpass_matched['hours'])
matched_df['state'] = gmaps_matched['state'].combine_first(overpass_matched['state'])
matched_df['url_gmaps'] = gmaps_matched['url']
matched_df['url_overpass'] = overpass_matched['url']
matched_df['avg_rating'] = gmaps_matched['avg_rating']
matched_df['num_of_reviews'] = gmaps_matched['num_of_reviews']
matched_df['price'] = gmaps_matched['price']
matched_df['misc'] = gmaps_matched['MISC']
matched_df['gmap_id'] = gmaps_matched['gmap_id']
matched_df['other_tags'] = overpass_matched['other_tags']
matched_df['fuente'] = 'gmap'

# 6. Datos solo en Google Maps
gmaps_indexes_matched = idx[matched_mask][:, 0]
gmaps_unmatched = gmaps_df.drop(gmaps_df.index[gmaps_indexes_matched]).reset_index(drop=True)

gmaps_only = pd.DataFrame()
gmaps_only['name'] = gmaps_unmatched['name']
gmaps_only['address'] = gmaps_unmatched['address']
gmaps_only['description_gmaps'] = gmaps_unmatched['description']
gmaps_only['description_overpass'] = None
gmaps_only['latitude'] = gmaps_unmatched['latitude']
gmaps_only['longitude'] = gmaps_unmatched['longitude']
gmaps_only['category'] = gmaps_unmatched['category']
gmaps_only['hours'] = gmaps_unmatched['hours']
gmaps_only['state'] = gmaps_unmatched['state']
gmaps_only['url_gmaps'] = gmaps_unmatched['url']
gmaps_only['url_overpass'] = None
gmaps_only['avg_rating'] = gmaps_unmatched['avg_rating']
gmaps_only['num_of_reviews'] = gmaps_unmatched['num_of_reviews']
gmaps_only['price'] = gmaps_unmatched['price']
gmaps_only['misc'] = gmaps_unmatched['MISC']
gmaps_only['gmap_id'] = gmaps_unmatched['gmap_id']
gmaps_only['other_tags'] = None
gmaps_only['fuente'] = 'gmap'

# 7. Datos solo en Overpass
overpass_unmatched = overpass_df[~matched_mask].reset_index(drop=True)

overpass_only = pd.DataFrame()
overpass_only['name'] = overpass_unmatched['name']
overpass_only['address'] = overpass_unmatched['address']
overpass_only['description_gmaps'] = None
overpass_only['description_overpass'] = overpass_unmatched['description']
overpass_only['latitude'] = overpass_unmatched['latitude']
overpass_only['longitude'] = overpass_unmatched['longitude']
overpass_only['category'] = overpass_unmatched['category']
overpass_only['hours'] = overpass_unmatched['hours']
overpass_only['state'] = overpass_unmatched['state']
overpass_only['url_gmaps'] = None
overpass_only['url_overpass'] = overpass_unmatched['url']
overpass_only['avg_rating'] = None
overpass_only['num_of_reviews'] = None
overpass_only['price'] = None
overpass_only['misc'] = None
overpass_only['gmap_id'] = None
overpass_only['other_tags'] = overpass_unmatched['other_tags']
overpass_only['fuente'] = 'overpass'

# 8. Concatenar todo
final_df = pd.concat([matched_df, gmaps_only, overpass_only], ignore_index=True)

final_df



#======================== Limpieza ========================#

final_df.info()


nulls = final_df.isnull().sum().sort_values(ascending=False)

# Convertir 'num_of_reviews' a numérico (forzando errores a NaN)
final_df['num_of_reviews'] = pd.to_numeric(final_df['num_of_reviews'], errors='coerce')


final_df['is_good_restaurant'] = (
    (final_df['avg_rating'] >= 4.2) &
    (final_df['num_of_reviews'] >= 30) &
    (final_df['price'].notnull()) &
    (final_df['hours'].notnull()) &
    (
        final_df['url_gmaps'].notnull() | final_df['url_overpass'].notnull()
    )
)
from shapely.geometry import Point
import geopandas as gpd

# Crear la geometría a partir de lat/long
final_df['geometry'] = final_df.apply(lambda row: Point(row['longitude'], row['latitude']), axis=1)

# Convertir a GeoDataFrame
gdf_all = gpd.GeoDataFrame(final_df, geometry='geometry', crs='EPSG:4326')



import matplotlib.pyplot as plt

# Cargar el mapa base
coord_nyc = gpd.read_file(path_coord_nyc)
coord_nyc = coord_nyc.to_crs(epsg=4326)

# Separar restaurantes buenos y no buenos
gdf_good = gdf_all[gdf_all['is_good_restaurant'] == True]
gdf_bad = gdf_all[gdf_all['is_good_restaurant'] == False]

# Plot
fig, ax = plt.subplots(figsize=(12, 12))
coord_nyc.plot(ax=ax, color='lightgrey', edgecolor='white')

# No buenos (rojo)
gdf_bad.plot(ax=ax, markersize=5, color='red', label='Otros', alpha=0.3)

# Buenos (verde)
gdf_good.plot(ax=ax, markersize=7, color='green', label='Buenos', alpha=1)


plt.title("Restaurantes Buenos vs Otros en NYC")
plt.legend()
plt.tight_layout()
plt.show()








# Rellenar valores faltantes para que no haya errores
final_df['avg_rating'] = final_df['avg_rating'].fillna(0)
final_df['num_of_reviews'] = final_df['num_of_reviews'].fillna(0)

# Normalizar valores para hacer un puntaje de 0 a 1 (puedes ajustar los pesos según tu criterio)
from sklearn.preprocessing import MinMaxScaler
import numpy as np

scaler = MinMaxScaler()

# Escalar valores
final_df[['rating_scaled', 'reviews_scaled']] = scaler.fit_transform(
    final_df[['avg_rating', 'num_of_reviews']]
)

# Puntaje compuesto (puedes ajustar los pesos)
final_df['score'] = (
    0.7 * final_df['rating_scaled'] +
    0.3 * final_df['reviews_scaled']
)

import matplotlib.pyplot as plt
import geopandas as gpd

# Convertir de nuevo a GeoDataFrame si hace falta
final_df['geometry'] = final_df.apply(lambda row: Point(row['longitude'], row['latitude']), axis=1)
gdf_all = gpd.GeoDataFrame(final_df, geometry='geometry', crs='EPSG:4326')

# Cargar mapa base
coord_nyc = gpd.read_file(path_coord_nyc)
coord_nyc = coord_nyc.to_crs(epsg=4326)

# Plot con hexbin (con valores del score)
fig, ax = plt.subplots(figsize=(12, 12))
coord_nyc.plot(ax=ax, color='lightgrey', edgecolor='white')

# Heatmap con hexbin
hb = ax.hexbin(
    gdf_all.geometry.x,
    gdf_all.geometry.y,
    C=gdf_all['score'],      # C para usar los valores de score
    gridsize=60,             # Puedes ajustar resolución
    reduce_C_function=np.mean,
    cmap='YlOrRd',           # Colores de calor
    mincnt=1,
    alpha=0.8
)

cb = fig.colorbar(hb, ax=ax, label='Puntaje promedio')
plt.title('Mapa de calor: Puntaje promedio de restaurantes en NYC')
plt.tight_layout()
plt.show()













#======================== Analisis ========================#

import matplotlib.pyplot as plt

# Conteo por fuente
conteo_fuente = final_df['fuente'].value_counts()

# Histograma
conteo_fuente.plot(kind='bar', color=['blue', 'red'], alpha=0.6)
plt.title('Cantidad de datos por fuente')
plt.xlabel('Fuente')
plt.ylabel('Cantidad de registros')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()

 
 
 
# ======================== mapa ========================#
path_coord_nyc = os.path.join(path_raw, 'coord_nyc/nybb.shp')

import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point

# Cargar mapa base de NYC y asegurarse que esté en EPSG:4326
coord_nyc = gpd.read_file(path_coord_nyc)
coord_nyc = coord_nyc.to_crs(epsg=4326)  # Convertimos si no está en el sistema correcto

# Crear GeoDataFrames para cada fuente
def crear_gdf(df):
    df = df.copy()
    df['geometry'] = df.apply(lambda row: Point(row['longitude'], row['latitude']), axis=1)
    return gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')

gdf_overpass = crear_gdf(final_df[final_df['fuente'] == 'overpass'])
gdf_gmap = crear_gdf(final_df[final_df['fuente'] == 'gmap'])

# Plot
fig, ax = plt.subplots(figsize=(12, 12))
coord_nyc.plot(ax=ax, color='lightgrey', edgecolor='white')
gdf_overpass.plot(ax=ax, markersize=5, color='blue', label='Overpass', alpha=0.6)
gdf_gmap.plot(ax=ax, markersize=5, color='red', label='Google Maps', alpha=0.6)

plt.title("Restaurantes NYC: Overpass (azul) vs GMaps (rojo)")
plt.legend()
plt.tight_layout()
plt.show()

#==================== analisis ====================#

final_df.info()

import seaborn as sns
import matplotlib.pyplot as plt

df = final_df.copy()


sns.scatterplot(data=df, x='longitude', y='latitude', hue='fuente', alpha=0.5, s=10)
plt.title('Distribución geográfica de restaurantes')
plt.xlabel('Longitud')
plt.ylabel('Latitud')
plt.legend()
plt.show()


top_categorias = df['category'].value_counts().nlargest(10)
sns.barplot(x=top_categorias.values, y=top_categorias.index, palette='magma')
plt.title('Top 10 categorías de restaurantes')
plt.xlabel('Cantidad')
plt.ylabel('Categoría')
plt.show()


sns.histplot(df['avg_rating'].dropna(), bins=20, kde=True, color='green')
plt.title('Distribución de calificaciones (GMaps)')
plt.xlabel('Rating promedio')
plt.ylabel('Cantidad')
plt.show()


(df.isnull().mean() * 100).sort_values(ascending=False).plot.barh(figsize=(10, 8), color='salmon')
plt.title('% de valores faltantes por columna')
plt.xlabel('% de NaNs')
plt.tight_layout()
plt.show()


df_valid = df[df['fuente'] == 'gmap'].copy()
df_valid['num_of_reviews'] = pd.to_numeric(df_valid['num_of_reviews'], errors='coerce')
sns.scatterplot(data=df_valid, x='avg_rating', y='num_of_reviews', alpha=0.4)
plt.title("Rating vs. Número de reseñas")
plt.xlabel("Rating promedio")
plt.ylabel("Número de reseñas")
plt.show()


import folium
from folium.plugins import HeatMap

# Filtrar restaurantes bien calificados
high_rated = df[(df['fuente'] == 'gmap') & (df['avg_rating'] >= 4.5)].copy()
# Crear mapa base centrado en NYC
nyc_center = [40.7128, -74.0060]
mapa = folium.Map(location=nyc_center, zoom_start=12, tiles='cartodbpositron')
# Agregar capa de calor
heat_data = high_rated[['latitude', 'longitude']].values.tolist()
HeatMap(heat_data, radius=10, blur=15, max_zoom=15).add_to(mapa)
# Mostrar mapa
mapa


