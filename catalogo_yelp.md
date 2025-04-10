# 📄 Catálogo de Datos - Yelp Dataset Unificado

Este dataset contiene información consolidada por estado a partir de los archivos `business.csv`, `review.csv`, `tip.csv`, `checkin.csv` y `user.csv`, con columnas traducidas al español y enriquecidas con análisis de sentimiento.Del la ciudad de new york.

---

## 🧱 Estructura del archivo

| Columna                    | Descripción                                                                 |
|----------------------------|-----------------------------------------------------------------------------|
| `id_negocio`               | Identificador único del negocio                                             |
| `nombre`                   | Nombre comercial del negocio                                                |
| `direccion`                | Dirección física del local                                                  |
| `ciudad`                   | Ciudad donde se ubica el negocio                                            |
| `estado`                   | Estado o provincia (abreviatura, ej. NY, CA)                                |
| `codigo_postal`            | Código postal                                                               |
| `latitud`                  | Coordenada geográfica (latitud)                                             |
| `longitud`                 | Coordenada geográfica (longitud)                                            |
| `categorias`               | Categorías del negocio separadas por comas                                  |
| `atributos`                | Atributos del negocio en formato diccionario (WiFi, estacionamiento, etc.) |
| `horario`                  | Horario de atención por día, en formato JSON                                |
| `estrellas`                | Calificación promedio (de 1 a 5)                                            |
| `cantidad_reviews`         | Número total de reseñas del negocio                                         |
| `esta_abierto`             | Si el negocio está abierto (1) o cerrado (0)                                |
| `fechas_visitas`           | Fechas de check-in (si existen)                                             |
| `texto_review_o_tip`       | Contenido textual de reseña o tip                                           |
| `estrellas_review`         | Puntuación dada en una reseña (1 a 5)                                       |
| `util_review`              | Votos que marcaron la reseña como útil                                      |
| `gracioso_review`          | Votos que marcaron la reseña como graciosa                                  |
| `cool_review`              | Votos que marcaron la reseña como interesante                               |
| `sentimiento`              | Score de sentimiento (VADER) entre -1 (negativo) y 1 (positivo)             |
| `tipo_sentimiento`         | Clasificación del sentimiento: positivo, neutro, negativo                   |
| `id_usuario`               | ID del usuario que escribió la reseña o tip                                 |
| `usuario_desde`            | Fecha en que el usuario se unió a Yelp                                      |
| `seguidores_usuario`       | Número de fans (seguidores) que tiene el usuario                            |
| `media_estrellas_usuario`  | Promedio de calificaciones otorgadas por el usuario                         |
| `cantidad_cumplidos_tip`   | Número de elogios recibidos en un tip                                       |

---

## 🔎 Nota

- Las columnas `sentimiento` y `tipo_sentimiento` fueron generadas usando **NLTK VADER**, un modelo especializado en análisis de texto informal.
- Algunas columnas pueden contener valores nulos si no existía información original (ej. horarios, atributos).


---
