import os
from datetime import datetime

def create_output_folder(base_dir):
    """Buat folder baru untuk setiap hasil deteksi"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = os.path.join(base_dir, f"run_{timestamp}")
    os.makedirs(folder, exist_ok=True)
    return folder
