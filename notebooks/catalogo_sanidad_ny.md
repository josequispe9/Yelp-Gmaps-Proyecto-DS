# 📘 Diccionario de Datos – Salubridad NY (Compatible con Yelp)

Este archivo contiene información sobre la inspección de restaurantes en la ciudad de Nueva York, estandarizado para integrarse con el dataset de Yelp.

---

| Columna              | Tipo     | Descripción                                                                 |
|----------------------|----------|-----------------------------------------------------------------------------|
| **nombre**           | Texto    | Nombre del restaurante. Coincide con el campo `nombre` del dataset Yelp.   |
| **direccion**        | Texto    | Dirección del restaurante. Coincide con `direccion` de Yelp.               |
| **ciudad**           | Texto    | Borough de NYC donde se ubica el local (ej. Manhattan, Brooklyn, etc.).    |
| **codigo_postal**    | Texto    | Código ZIP del restaurante.                                                |
| **latitud**          | Float    | Coordenada geográfica del local.                                           |
| **longitud**         | Float    | Coordenada geográfica del local.                                           |
| **estado_cumplimiento** | Texto | Estado de cumplimiento de las normativas (ej. `Compliant`, `Non-Compliant`).|
| **fecha_inspeccion** | Texto / Timestamp | Fecha y hora en que se realizó la inspección.                        |
| **puntos_violacion** | Entero   | Puntuación asignada por violaciones encontradas.                          |
| **codigo_violacion** | Entero   | Código asociado a la infracción.                                           |

---

## 🧠 Nota

- Los nombres de columnas coinciden con los del dataset **Yelp** para permitir integraciones y cruces (`merge`, `join`, `concat`).
- Puedes usar campos como `nombre`, `direccion`, y `codigo_postal` como claves para el emparejamiento.
- Los campos `latitud` y `longitud` permiten análisis geoespacial o visualización en mapas.
- El campo `estado_cumplimiento` es clave para evaluar condiciones sanitarias.



