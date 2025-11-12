import cv2
import yt_dlp
import os
import time
import numpy as np
from ultralytics import YOLO
import torch 
import mysql.connector
import shutil  # <-- Penting untuk ZIP

# --- Konstanta dan Mapping ---
TARGET_WIDTH = 640 
TARGET_HEIGHT = 360 
LINE_Y = int(TARGET_HEIGHT * 0.5)
COUNT_TOLERANCE = 10 

# --- (Model 1) Model Inisialisasi (Gunakan Model Berbeda) ---
MODEL_NAME = 'yolov8n.pt' # <-- Menggunakan model standar YOLOv8 Nano (COCO)
try:
    PROCESS_MODEL = YOLO(MODEL_NAME)
    DEVICE = '0' if torch.cuda.is_available() else 'cpu'
    USE_HALF = torch.cuda.is_available()
    print(f"✅ PROCESSOR: Model BARU ({MODEL_NAME}) siap diaktifkan pada {DEVICE}.")
    PROCESSOR_CLASS_NAMES = PROCESS_MODEL.model.names
except Exception as e:
    print(f"🔥 PROCESSOR ERROR: Gagal memuat model YOLO: {e}")
    PROCESS_MODEL = None
    PROCESSOR_CLASS_NAMES = {}

# --- (Model 2) Mapping dari NAMA KELAS Model Baru ke Database Anda ---
PROCESSOR_DB_MAPPING = {
    'car': 'total_car',
    'motorcycle': 'total_motorcycle',
    'bus': 'total_bus',
    'truck': 'total_truck'
}

# --- Fungsi Database (Independent) ---
def get_db_connection(db_config):
    try:
        conn = mysql.connector.connect(**db_config)
        return conn
    except mysql.connector.Error as err:
        print(f"🔥 PROCESSOR DB ERROR: Gagal koneksi: {err}")
        return None

def update_db_count(db_config, column_name):
    conn = get_db_connection(db_config)
    if conn:
        try:
            cursor = conn.cursor()
            query = f"UPDATE traffic_stats SET {column_name} = {column_name} + 1 WHERE id = 1"
            cursor.execute(query)
            conn.commit()
            print(f"✅ PROCESSOR DB: Berhasil menambah 1 ke {column_name}")
        except mysql.connector.Error as err:
            print(f"🔥 PROCESSOR DB ERROR: Gagal update: {err}")
        finally:
            if conn and conn.is_connected():
                cursor.close()
                conn.close()

# --- Fungsi Pengambilan Stream YouTube ---
def get_youtube_stream_url(url, quality='480p'):
    try:
        ydl_opts = {'format': 'bestvideo[height<=?480]+bestaudio/best', 'quiet': True, 'skip_download': True,}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=False)
            stream_url = info_dict.get('url', info_dict.get('formats')[0]['url'])
            return stream_url
    except Exception:
        return None

# --- Fungsi Utama Pemrosesan (ZIP) ---
def process_video_task(task_id, source_type, source_path, db_config):
    if not PROCESS_MODEL:
        print(f"🔥 TASK {task_id}: Gagal karena model tidak termuat.")
        return

    print(f"🚀 TASK {task_id}: Memulai pemrosesan dengan model {MODEL_NAME} dari {source_type}")

    if source_type == 'youtube':
        video_source = get_youtube_stream_url(source_path) or source_path
        output_base_path = 'uploads'
    else: # 'file'
        video_source = source_path
        output_base_path = os.path.dirname(source_path)
        
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"🔥 TASK {task_id}: Gagal membuka sumber video.")
        return

    # --- BUAT FOLDER PENYIMPANAN GAMBAR ---
    task_temp_dir = os.path.join(output_base_path, f"processed_{task_id}_temp")
    if not os.path.exists(task_temp_dir):
        os.makedirs(task_temp_dir)
    
    tracked_vehicles = {} 
    frame_save_count = 0 
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))
        annotated_frame = frame.copy()
        
        results = PROCESS_MODEL.track(frame, persist=True, verbose=False, imgsz=TARGET_WIDTH, device=DEVICE, half=USE_HALF) 

        cv2.line(annotated_frame, (0, LINE_Y), (TARGET_WIDTH, LINE_Y), (0, 0, 255), 2)
        
        has_relevant_detections = False # Flag untuk menandai apakah frame ini perlu disimpan

        if results and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            
            for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                x1, y1, x2, y2 = box
                centroid_y = (y1 + y2) // 2
                
                vehicle_name = PROCESSOR_CLASS_NAMES.get(class_id, "Unknown")
                db_column = PROCESSOR_DB_MAPPING.get(vehicle_name, None)
                
                if db_column: 
                    has_relevant_detections = True 
                
                if track_id not in tracked_vehicles:
                    tracked_vehicles[track_id] = {'class': vehicle_name, 'counted': False}
                
                is_passing_line = (centroid_y > LINE_Y - COUNT_TOLERANCE) and \
                                  (centroid_y < LINE_Y + COUNT_TOLERANCE)
                
                if is_passing_line and not tracked_vehicles[track_id]['counted']:
                    if db_column:
                        update_db_count(db_config, db_column) 
                    tracked_vehicles[track_id]['counted'] = True
                    cv2.line(annotated_frame, (0, LINE_Y), (TARGET_WIDTH, LINE_Y), (0, 255, 255), 4)
                
                if db_column and not tracked_vehicles[track_id]['counted']:
                    label = f"{vehicle_name} ID:{track_id}"
                    cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # --- SIMPAN FRAME SEBAGAI JPG ---
        if has_relevant_detections:
            frame_filename = os.path.join(task_temp_dir, f"frame_{frame_save_count:05d}.jpg")
            cv2.imwrite(frame_filename, annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            frame_save_count += 1

    cap.release()

    print(f"🏁 TASK {task_id}: Pemrosesan video selesai. {frame_save_count} gambar anotasi disimpan.")

    # --- BUAT FILE ZIP DAN HAPUS FOLDER TEMP ---
    if frame_save_count > 0:
        print(f"📦 TASK {task_id}: Membuat file ZIP...")
        zip_output_path_base = os.path.join(output_base_path, f"processed_{task_id}")
        
        try:
            shutil.make_archive(
                base_name=zip_output_path_base, 
                format='zip',                   
                root_dir=task_temp_dir          
            )
            print(f"✅ TASK {task_id}: File ZIP dibuat di {zip_output_path_base}.zip")
            
            shutil.rmtree(task_temp_dir)
            print(f"🗑️ TASK {task_id}: Folder gambar sementara {task_temp_dir} dihapus.")
            
        except Exception as e:
            print(f"🔥 ERROR ZIP: Gagal membuat arsip ZIP atau menghapus folder. Error: {e}")
            
    else:
        print(f"ℹ️ TASK {task_id}: Tidak ada deteksi, file ZIP tidak dibuat.")