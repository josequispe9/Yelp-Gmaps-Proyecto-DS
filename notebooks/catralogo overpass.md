## 📚 Diccionario de Datos – Overpass Turbo (`export.json`)

Este documento describe la estructura y significado de los campos del archivo `export.json`, obtenido a partir de **Overpass Turbo** sobre datos de OpenStreetMap (OSM), centrados en restaurantes del estado de Nueva York.

---

### 💊 Estructura General

- **Archivo**: `export.json`
- **Estructura raíz**: Objeto JSON con clave principal `elements` (lista de nodos)
- Cada `element` contiene:
  - `type`: siempre "node" (tipo de objeto OSM)
  - `id`: identificador único en OSM
  - `lat`: latitud (coordenadas)
  - `lon`: longitud (coordenadas)
  - `tags`: diccionario de metadatos

---

### 📂 Campos dentro de `tags`

| Campo                | Descripción                                                                 | Tipo     | Ejemplo                              |
|---------------------|----------------------------------------------------------------------------------|----------|--------------------------------------|
| `name`              | Nombre del establecimiento                                                      | Texto    | `"Jim's Steak Out"`                 |
| `amenity`           | Tipo de servicio (ej: `restaurant`, `cafe`, `fast_food`)                        | Texto    | `"restaurant"`                      |
| `cuisine`           | Tipo de comida servida (puede ser lista separada por `;`)                        | Texto    | `"mexican"` o `"pizza;burger"`     |
| `addr:housenumber`  | Número de casa o local                                                       | Texto    | `"194"`                             |
| `addr:street`       | Nombre de la calle                                                             | Texto    | `"Allen Street"`                    |
| `addr:city`         | Ciudad donde se encuentra el local                                             | Texto    | `"Buffalo"`                         |
| `addr:postcode`     | Código postal del establecimiento                                              | Texto    | `"14201"`                           |
| `addr:state`        | Estado (por defecto: "NY")                                                     | Texto    | `"NY"`                              |
| `phone`             | Teléfono del establecimiento                                                  | Texto    | `"+17168862222"`                    |
| `website`           | URL del sitio web oficial                                                      | URL      | `"https://www.jimssteakout.com"`   |
| `brand`             | Marca comercial (si aplica)                                                    | Texto    | `"KFC"`                             |
| `opening_hours`     | Horario de atención (formato OSM)                                               | Texto    | `"Mo-Su 10:00-22:00"`               |
| `internet_access`   | Acceso a internet (`yes`, `no`, `wlan`)                                        | Texto    | `"wlan"`                            |
| `takeaway`          | Comida para llevar (`yes` / `no`)                                              | Texto    | `"yes"`                             |
| `wheelchair`        | Accesibilidad para sillas de ruedas                                            | Texto    | `"yes"` / `"no"`                   |
| `contact:facebook`  | Enlace a página de Facebook                                                   | URL      | `"https://facebook.com/..."`        |
| `contact:website`   | Sitio web oficial alternativo                                                  | URL      | `"https://..."`                     |
| `created_by`        | Software con el que se creó el nodo                                            | Texto    | `"Potlatch 0.9a"`                   |

---

### 🌎 Coordenadas

| Campo  | Descripción                    | Tipo   | Ejemplo        |
|--------|----------------------------------|--------|----------------|
| `lat`  | Latitud geográfica               | Float  | `42.8993729`   |
| `lon`  | Longitud geográfica              | Float  | `-78.8772291`  |

---

### 📈 Nota

- No todos los campos están presentes en todos los nodos.
- `cuisine` puede contener múltiples valores separados por punto y coma.
- Es recomendable **normalizar el campo `name`** (minúsculas, sin tildes, sin espacios dobles) antes de hacer merge o generar claves de dimensión.

---


