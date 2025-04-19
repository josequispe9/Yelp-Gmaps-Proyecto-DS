import os

# Nombre del proyecto y carpeta compartida
PROJECT_FOLDER = 'Project_gourmet_advisor'
SHARED_DRIVE_SEGMENT = 'Unidades compartidas'
DATA_FOLDER = 'datalake'

def find_shared_drive_path(project_name=PROJECT_FOLDER, drive_segment=SHARED_DRIVE_SEGMENT, data_folder=DATA_FOLDER):
    """
    Busca en todas las unidades montadas hasta encontrar la ruta del proyecto compartido.
    """
    possible_drives = [f"{d}:\\" for d in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' if os.path.exists(f"{d}:\\")]
    for drive in possible_drives:
        candidate_path = os.path.join(drive, drive_segment, project_name, data_folder)
        if os.path.exists(candidate_path):
            return candidate_path
    return None

def get_data_paths():
    """
    Devuelve las rutas a las subcarpetas 'raw', 'processed', 'interim', 'external' dentro de 'datalake'.
    """
    base_drive_path = find_shared_drive_path()
    if base_drive_path is None:
        raise FileNotFoundError("No se encontró la carpeta del proyecto en Google Drive.")

    return {
        'base': base_drive_path,
        'raw': os.path.join(base_drive_path, 'raw'),
        'processed': os.path.join(base_drive_path, 'processed'),
        'interim': os.path.join(base_drive_path, 'interim'),
        'external': os.path.join(base_drive_path, 'external'),
    }



### ------------------ Ejemplo de uso ------------------ ###

"""
from path_manager import get_data_paths

paths = get_data_paths()
print(paths['raw'])
print(paths['processed'])

"""

