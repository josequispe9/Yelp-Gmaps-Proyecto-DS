import os

# Nombre de tu carpeta de proyecto principal en Drive
project_folder = 'Project_gourmet_advisor'

# Detecta automáticamente el sistema operativo y define la ruta base
if os.name == 'nt':  # Windows
    base_drive_path = os.path.join(os.environ['USERPROFILE'], 'Google Drive', project_folder, 'data')
else:  # macOS/Linux
    base_drive_path = os.path.expanduser(f'~/Google Drive/{project_folder}/data')

# Subcarpetas dentro de "data"
raw_path = os.path.join(base_drive_path, 'raw')
processed_path = os.path.join(base_drive_path, 'processed')
interim_path = os.path.join(base_drive_path, 'interim')
external_path = os.path.join(base_drive_path, 'external')
