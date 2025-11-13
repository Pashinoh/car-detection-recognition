# app.py (KODE INI MENGATASI MASALAH IMPOR BERAT DAN BATAS THREAD)

# ====================================================================
# 1. PERBAIKAN LIMIT SUMBER DAYA (PASTIKAN INI DI BARIS ATAS!)
# Ini mencegah error OpenBLAS: 'Resource temporarily unavailable'
# ====================================================================
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

# ====================================================================
# 2. IMPOR RINGAN (TIDAK MEMUAT cv2, torch, numpy, YOLO)
# ====================================================================
import yt_dlp
import uuid
import threading
import time
from datetime import datetime, timedelta
from flask import Flask, Response, render_template, jsonify, request, redirect, url_for, send_file
import mysql.connector
from apscheduler.schedulers.background import BackgroundScheduler
import io  # Untuk CSV
import csv # Untuk CSV

# Import processor (Logika deteksi untuk tugas latar belakang)
try:
    from processor import process_video_task 
except ImportError:
    print("PERINGATAN: Gagal mengimpor 'processor.py'. Pastikan file tersebut ada.")
    process_video_task = None

# ====================================================================
# 3. VARIABEL STATUS MODEL GLOBAL
# ====================================================================
HEAVY_DEPS_LOADED = False
# Inisiasi dengan None, akan diisi oleh load_heavy_dependencies()
MODEL = None 
DEVICE = 'cpu'
USE_HALF = False
cv2 = None # Variabel untuk cv2 global
np = None # Variabel untuk numpy global

# --- Konfigurasi Database ---
DB_CONFIG = {
    'user': 'root',       
    'password': '',       
    'host': '127.0.0.1',  
    'database': 'traffic_db' 
}

# --- Konfigurasi Global (Bukan Konstanta) ---
CURRENT_YOUTUBE_URL = "https://www.youtube.com/live/aV0fzw8wH2o?si=f-nuqF2KzWUQ0tsP" 
MODEL_NAME = 'model/best.pt' # <-- Model CUSTOM Anda untuk Live Stream
CCTV_LOCATION = "Jembatan (DB Connected)" 
VEHICLE_CLASSES = [0, 1, 2, 3] # <-- ID Kelas Custom

# RESOLUSI & GARIS
TARGET_WIDTH = 640 
TARGET_HEIGHT = 360 
LINE_Y = int(TARGET_HEIGHT * 0.5) 
COUNT_TOLERANCE = 10 
LINE_START = (0, LINE_Y)
LINE_END = (TARGET_WIDTH, LINE_Y)

FRAME_SKIP = 1 
MAX_TRACKING_AGE = 30 
frame_counter = 0

# Mapping class IDs ke nama (HANYA UNTUK LIVE STREAM)
CLASS_MAPPING = {
    0: ('Car', 'total_car'),
    1: ('Motorcycle', 'total_motorcycle'),
    2: ('Bus', 'total_bus'),
    3: ('Truck', 'total_truck')
}

# --- Variabel Global untuk Tracking & FPS ---
tracked_vehicles = {} 
current_fps = 0.0 
latest_results = None 
db_connected = False

# --- Konfigurasi Flask & Folder Upload ---
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
    
# ====================================================================
# 4. INISIASI FLASK (PASTIKAN INI TERJADI SETELAH IMPOR RINGAN)
# ====================================================================
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['TEMPLATES_AUTO_RELOAD'] = True

scheduler = BackgroundScheduler(daemon=True)
scheduler.start()

# ====================================================================
# 5. FUNGSI UNTUK MEMUAT DEPENDENSI BERAT (LAZY LOADING)
# ====================================================================
def load_heavy_dependencies():
    """Memuat model dan pustaka berat (cv2, torch, numpy, YOLO) saat dibutuhkan."""
    global HEAVY_DEPS_LOADED, MODEL, DEVICE, USE_HALF, cv2, np
    
    if HEAVY_DEPS_LOADED:
        return True

    print("INFO: Mencoba memuat dependensi berat (cv2, torch, YOLO)...")
    try:
        # Pindahkan impor berat ke sini!
        import cv2 as cv2_local
        import numpy as np_local
        import torch
        from ultralytics import YOLO

        cv2 = cv2_local # Alihkan variabel global
        np = np_local # Alihkan variabel global

        # 1. Tentukan Perangkat
        if torch.cuda.is_available():
            DEVICE = '0' 
            USE_HALF = True 
            print("GPU CUDA terdeteksi. Inferensi akan menggunakan GPU.")
        else:
            DEVICE = 'cpu'
            USE_HALF = False
            print("GPU CUDA tidak terdeteksi. Inferensi akan menggunakan CPU.")

        # 2. Muat Model
        MODEL = YOLO(MODEL_NAME)
        print(f"Model Live Stream ({MODEL_NAME}) berhasil dimuat.")
        
        HEAVY_DEPS_LOADED = True
        return True

    except Exception as e:
        # PENTING: Jika gagal, aplikasi tetap boot, tetapi video feed akan menunjukkan error
        print(f"FATAL ERROR: Gagal memuat dependensi berat. Live feed akan non-aktif. Error: {e}")
        HEAVY_DEPS_LOADED = False
        return False


# --- Fungsi Database ---
def get_db_connection():
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        global db_connected
        db_connected = True
        return conn
    except mysql.connector.Error as err:
        db_connected = False
        return None

def update_db_count(column_name):
    """Menambah +1 ke tabel TOTAL (traffic_stats)"""
    conn = get_db_connection()
    if conn:
        try:
            cursor = conn.cursor()
            query = f"UPDATE traffic_stats SET {column_name} = {column_name} + 1 WHERE id = 1"
            cursor.execute(query)
            conn.commit()
            print(f"DB INFO (Live): Menambah 1 ke {column_name}")
        except mysql.connector.Error as err:
            print(f"ERROR DB: Gagal update total: {err}")
        finally:
            if conn and conn.is_connected():
                cursor.close()
                conn.close()

def log_detection_to_db(ts, class_name, track_id, source_id):
    """Menyisipkan log deteksi BARU ke tabel detection_log"""
    conn = get_db_connection()
    if conn:
        try:
            cursor = conn.cursor()
            query = """
                INSERT INTO detection_log (timestamp, class_name, track_id, source)
                VALUES (%s, %s, %s, %s)
            """
            values = (ts, class_name, track_id, source_id)
            cursor.execute(query, values)
            conn.commit()
            print(f"DB LOG (Live): Mencatat {class_name} (ID: {track_id})")
        except mysql.connector.Error as err:
            print(f"ERROR DB: Gagal mencatat log: {err}")
        finally:
            if conn and conn.is_connected():
                cursor.close()
                conn.close()
# -----------------------------------------------

def fetch_db_counts():
    """Mengambil TOTAL hitungan dari traffic_stats"""
    conn = get_db_connection()
    if conn:
        try:
            cursor = conn.cursor(dictionary=True)
            query = "SELECT * FROM traffic_stats WHERE id = 1"
            cursor.execute(query)
            result = cursor.fetchone()
            return result
        except mysql.connector.Error as err:
            return None
        finally:
            if conn and conn.is_connected():
                cursor.close()
                conn.close()
    return None

def reset_db_counts():
    """(DIPERBARUI) Mengatur ulang KEDUA tabel (total dan log)"""
    conn = get_db_connection()
    if conn:
        try:
            cursor = conn.cursor()
            # 1. Reset tabel total
            query_total = """
                UPDATE traffic_stats 
                SET total_car = 0, 
                    total_motorcycle = 0, 
                    total_bus = 0, 
                    total_truck = 0 
                WHERE id = 1
            """
            cursor.execute(query_total)
            
            # 2. Hapus semua data dari tabel log
            query_log = "TRUNCATE TABLE detection_log"
            cursor.execute(query_log)
            
            conn.commit()
            print("DATABASE RESET: Tabel 'traffic_stats' dan 'detection_log' telah direset.")
        except mysql.connector.Error as err:
            print(f"ERROR: Gagal me-reset database: {err}")
        finally:
            if conn and conn.is_connected():
                cursor.close()
                conn.close()

# --- Fungsi Pengambilan Stream YouTube ---
def get_youtube_stream_url(url, quality='480p'):
    try:
        ydl_opts = {'format': f'bestvideo[height<=?{quality[:-1]}]+bestaudio/best', 'quiet': True, 'skip_download': True,}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=False)
            # Menggunakan 'url' atau format pertama yang tersedia
            stream_url = info_dict.get('url', info_dict.get('formats')[0]['url'])
            return stream_url
    except Exception as e:
        print(f"ERROR YT-DLP: Gagal mendapatkan URL stream: {e}")
        return None

# --- Fungsi Deteksi (LIVE STREAM) ---
def generate_frames():
    global tracked_vehicles, current_fps, frame_counter, latest_results, CURRENT_YOUTUBE_URL
    
    # 1. PANGGIL LOADER DI SINI!
    if not HEAVY_DEPS_LOADED:
        if not load_heavy_dependencies():
            # Jika loading gagal, kirim frame error
            print("Stream Gagal: Dependensi tidak termuat.")
            
            # Buat frame kosong menggunakan variabel global cv2/np
            if cv2 and np:
                # PASTIKAN cv2 DAN np SUDAH DIIMPOR, jika tidak, Anda akan mendapat error saat membuat frame.
                frame = np.zeros((TARGET_HEIGHT, TARGET_WIDTH, 3), dtype=np.uint8) 
                cv2.putText(frame, "MODEL GAGAL DIMUAT (CEK LOG SERVER)", (50, TARGET_HEIGHT // 2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 40])
                if ret:
                    yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            return

    # KODE HANYA BERJALAN JIKA MODEL SUKSES DIMUAT
    cap = None
    active_url = CURRENT_YOUTUBE_URL 
    
    while cap is None or not cap.isOpened():
        stream_url = get_youtube_stream_url(active_url) or active_url
        cap = cv2.VideoCapture(stream_url) # Gunakan cv2 global
        if not cap.isOpened():
            print("FATAL ERROR: Gagal membuka stream video. Mencoba lagi dalam 5 detik...")
            current_fps = 0.0
            time.sleep(5)
            if active_url != CURRENT_YOUTUBE_URL:
                print("INFO: URL Stream diganti, menghentikan percobaan koneksi lama.")
                break
            continue
        print(f"INFO: Stream berhasil dibuka untuk {active_url}")

    while True:
        if active_url != CURRENT_YOUTUBE_URL:
            print("INFO: URL Stream diganti. Menghentikan stream lama.")
            cap.release()
            break
            
        start_time = time.time() 
        ret, frame = cap.read()
        if not ret:
            print("WARNING: Frame gagal dibaca. Mencoba membuka ulang stream...")
            cap.release()
            break 
            
        frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT)) # Gunakan cv2 global
        
        # Ganti model.track menjadi MODEL.track (MODEL adalah variabel global)
        results = MODEL.track(frame, persist=True, classes=VEHICLE_CLASSES, verbose=False, imgsz=TARGET_WIDTH, device=DEVICE, half=USE_HALF) 
        latest_results = results 
        
        if results and results[0].boxes.id is not None:
            current_ids = set(results[0].boxes.id.cpu().numpy().astype(int))
            for tid in current_ids:
                if tid in tracked_vehicles:
                    tracked_vehicles[tid]['age'] = 0 
        
        annotated_frame = frame.copy()
        cv2.line(annotated_frame, LINE_START, LINE_END, (0, 0, 255), 2) # Gunakan cv2 global
        
        if latest_results and latest_results[0].boxes.id is not None:
            boxes = latest_results[0].boxes.xyxy.cpu().numpy().astype(int)
            track_ids = latest_results[0].boxes.id.cpu().numpy().astype(int)
            class_ids = latest_results[0].boxes.cls.cpu().numpy().astype(int)
            new_tracked_vehicles = {}
            
            for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                x1, y1, x2, y2 = box
                centroid_y = (y1 + y2) // 2
                vehicle_name, db_column = CLASS_MAPPING.get(class_id, ('Unknown', None))
                
                if track_id not in tracked_vehicles:
                    tracked_vehicles[track_id] = {'class': vehicle_name, 'counted': False, 'age': 0}
                
                is_passing_line = (centroid_y > LINE_Y - COUNT_TOLERANCE) and (centroid_y < LINE_Y + COUNT_TOLERANCE)
                
                if is_passing_line and not tracked_vehicles[track_id]['counted']:
                    if db_column:
                        # 1. Update total hitungan (Tabel Lama)
                        update_db_count(db_column) 
                        
                        # 2. Catat log timestamp (Tabel Baru)
                        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        log_detection_to_db(ts, vehicle_name, int(track_id), "Live Stream")
                        
                    tracked_vehicles[track_id]['counted'] = True
                    cv2.line(annotated_frame, LINE_START, LINE_END, (0, 255, 255), 4) # Gunakan cv2 global
                
                if not tracked_vehicles[track_id]['counted']:
                    label = f"{vehicle_name} ID:{track_id}"
                    cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2) # Gunakan cv2 global
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2) # Gunakan cv2 global
                new_tracked_vehicles[track_id] = tracked_vehicles[track_id]

            for tid in tracked_vehicles:
                if tid not in new_tracked_vehicles:
                    tracked_vehicles[tid]['age'] += 1
            keys_to_delete = [tid for tid, data in tracked_vehicles.items() if (data['counted'] and data['age'] > MAX_TRACKING_AGE)]
            for tid in keys_to_delete:
                if tid in tracked_vehicles:
                    del tracked_vehicles[tid]
            
        curr_time = time.time()
        current_fps = 1 / (curr_time - start_time) if (curr_time - start_time) > 0 else 0 
        
        status_text = 'DB CONNECTED' if db_connected else 'DB OFFLINE'
        status_color = (0, 255, 0) if db_connected else (0, 0, 255) 
        cv2.putText(annotated_frame, f"FPS: {current_fps:.2f} | {status_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2) # Gunakan cv2 global

        ret, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 40]) # Gunakan cv2 global
        if not ret:
            continue
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# --- Fungsi Hapus File Terjadwal (TIDAK BERUBAH) ---
def delete_file(filepath):
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
            print(f"SCHEDULER: File {filepath} berhasil dihapus.")
        else:
            print(f"INFO SCHEDULER: File {filepath} tidak ditemukan (mungkin sudah dihapus).")
    except Exception as e:
        print(f"ERROR SCHEDULER: Gagal menghapus {filepath}. Error: {e}")

# ----------------------------------------------------
#               DEFINISI ROUTES (TIDAK BERUBAH)
# ----------------------------------------------------

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/analytics_data')
def analytics_data():
    """Endpoint untuk data statistik DB real-time (TOTAL)."""
    global current_fps, CCTV_LOCATION, db_connected, CURRENT_YOUTUBE_URL
    db_counts = fetch_db_counts() # Mengambil dari traffic_stats (Total)
    
    counts_data = {'Total': 0, 'Car': 0, 'Motorcycle': 0, 'Bus': 0, 'Truck': 0}
    if db_counts:
        # Gunakan get untuk menghindari KeyError jika kolom belum ada di DB
        counts_data['Car'] = db_counts.get('total_car', 0)
        counts_data['Motorcycle'] = db_counts.get('total_motorcycle', 0)
        counts_data['Bus'] = db_counts.get('total_bus', 0)
        counts_data['Truck'] = db_counts.get('total_truck', 0)
        counts_data['Total'] = sum(v for k, v in counts_data.items() if k != 'Total')

    response_data = {
        'location': CCTV_LOCATION,
        'fps': f"{current_fps:.2f}",
        'db_status': 'ONLINE' if db_connected and db_counts is not None else 'OFFLINE',
        'counts': counts_data, 
        'current_url': CURRENT_YOUTUBE_URL 
    }
    return jsonify(response_data)

@app.route('/')
def index():
    return render_template('index.html', current_url=CURRENT_YOUTUBE_URL)

@app.route('/update_stream', methods=['POST'])
def update_stream():
    """(DIPERBARUI) Mengganti URL dan MERESET KEDUA tabel database."""
    global CURRENT_YOUTUBE_URL, tracked_vehicles, current_fps
    
    if request.method == 'POST':
        new_url = request.form.get('youtube_url')
        if new_url and ("youtube.com" in new_url or "youtu.be" in new_url):
            CURRENT_YOUTUBE_URL = new_url
            print(f"STREAM UPDATE: URL diubah menjadi {new_url}")
            
            # Reset KEDUA tabel (total dan log)
            reset_db_counts()
            
            tracked_vehicles.clear()
            current_fps = 0.0
        else:
            print("WARNING: Menerima URL tidak valid, diabaikan.")
            
    return redirect(url_for('index'))

@app.route('/upload', methods=['GET', 'POST'])
def upload_page():
    if request.method == 'POST':
        # PENTING: Jika processor.py gagal diimpor, tampilkan error
        if process_video_task is None:
            return render_template('upload.html', error="ERROR: 'processor.py' tidak ditemukan atau gagal dimuat.")
            
        youtube_url = request.form.get('youtube_url')
        video_file = request.files.get('video_file')
        
        task_id = str(uuid.uuid4())
        filepath = None
        source_path = None
        source_type = None
        
        if video_file and video_file.filename != '':
            filename = f"{task_id}_{video_file.filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            video_file.save(filepath)
            source_type = 'file'
            source_path = filepath
        
        elif youtube_url:
            source_type = 'youtube'
            source_path = youtube_url
        else:
            return render_template('upload.html', error="Pilih file atau masukkan URL YouTube.")

        # Panggil fungsi dari processor.py di thread baru
        thread = threading.Thread(
            target=process_video_task, 
            args=(task_id, source_type, source_path, DB_CONFIG),
            daemon=True
        )
        thread.start()

        processed_filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"processed_{task_id}.zip") 
        delete_time = datetime.now() + timedelta(minutes=15) 

        if filepath: # Jadwalkan hapus file SUMBER
            scheduler.add_job(
                delete_file, 'date', 
                run_date=delete_time, 
                args=[filepath], 
                id=f"delete_src_{task_id}", 
                replace_existing=True
            )
            print(f"⏰ SCHEDULER: File SUMBER {filepath} akan dihapus pada {delete_time}")

        # Jadwalkan hapus file HASIL (jaring pengaman .zip)
        scheduler.add_job(
            delete_file, 'date', 
            run_date=delete_time, 
            args=[processed_filepath], 
            id=f"delete_proc_{task_id}", 
            replace_existing=True
        )
        print(f"⏰ SCHEDULER: File HASIL {processed_filepath} akan dihapus pada {delete_time} (jika tidak diunduh)")

        return redirect(url_for('task_status', task_id=task_id))

    return render_template('upload.html')

@app.route('/status/<task_id>')
def task_status(task_id):
    download_path = os.path.join(app.config['UPLOAD_FOLDER'], f"processed_{task_id}.zip") 
    is_ready = os.path.exists(download_path)
    status_msg = "Pemrosesan Selesai! File siap diunduh." if is_ready else "Sedang diproses. Mohon tunggu..."
        
    return render_template('task_status.html', task_id=task_id, status=status_msg, is_ready=is_ready)

@app.route('/download/<task_id>')
def download_file(task_id):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"processed_{task_id}.zip") 
    
    if os.path.exists(filepath):
        # Perpanjang waktu hapus setelah diunduh (5 menit)
        delete_time = datetime.now() + timedelta(minutes=5) 
        
        scheduler.add_job(
            delete_file, 'date', 
            run_date=delete_time, 
            args=[filepath], 
            id=f"delete_proc_{task_id}", 
            replace_existing=True
        )
        
        print(f"⏰ SCHEDULER: File HASIL {filepath} akan dihapus pada {delete_time} (Jadwal diperbarui)")
        
        return send_file(
            filepath, 
            mimetype='application/zip',
            as_attachment=True, 
            download_name=f"processed_traffic_{task_id}.zip"
        )
    
    return "File tidak ditemukan atau tidak ada deteksi (file ZIP tidak dibuat).", 404

# ----------------------------------------------------
#               (RUTE DOWNLOAD CSV)
# ----------------------------------------------------
@app.route('/download_rekap_csv')
def download_rekap_csv():
    """Mengunduh log deteksi mentah dari tabel 'detection_log'"""
    
    conn = get_db_connection()
    if not conn:
        return "Database error.", 500
        
    try:
        cursor = conn.cursor(dictionary=True)
        # 1. Ambil SEMUA data dari tabel log baru
        query = "SELECT id, timestamp, class_name, track_id, source FROM detection_log ORDER BY timestamp ASC"
        cursor.execute(query)
        log_data = cursor.fetchall()
        
        if not log_data:
            return "Tidak ada data log untuk diekspor (database kosong).", 404

        # 2. Siapkan file CSV di memori
        output = io.StringIO()
        fieldnames = ['id', 'timestamp', 'class_name', 'track_id', 'source']
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        
        # 3. Tulis data ke CSV
        writer.writeheader() 
        writer.writerows(log_data) 
        
        # 4. Siapkan file untuk didownload oleh user
        output.seek(0)
        filename = f"rekap_deteksi_kendaraan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        return Response(
            output.getvalue(),
            mimetype="text/csv",
            headers={
                "Content-Disposition": f"attachment;filename={filename}"
            }
        )
        
    except mysql.connector.Error as err:
        return f"Error saat mengambil data log: {err}", 500
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()

# --- Jalankan Aplikasi ---
if __name__ == '__main__':
    print("="*50)
    print("  Aplikasi Web Analisis Lalu Lintas Siap.")
    print("  Akses dashboard live di http://127.0.0.1:5000/")
    print("  Akses halaman upload di http://127.0.0.1:5000/upload")
    print("="*50)
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True, use_reloader=False)