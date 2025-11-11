import cv2
import yt_dlp
from ultralytics import YOLO
import time
from flask import Flask, Response, render_template, jsonify
import numpy as np
import torch # Import PyTorch untuk pengecekan ketersediaan GPU

# --- Konfigurasi Utama untuk GPU dan Akurasi ---
YOUTUBE_URL = "https://www.youtube.com/live/3BvTMxI14wg?si=aD-giLwax1qzGVqd" 
MODEL_NAME = 'yolov8n.pt'  # Model Small (Akurasi Lebih Baik)
VEHICLE_CLASSES = [2, 3, 5, 7] # 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'

# RESOLUSI AGAK BAGUS: 640x360
TARGET_WIDTH = 640 
TARGET_HEIGHT = 360 
LINE_Y = int(TARGET_HEIGHT * 0.6) 
LINE_START = (0, LINE_Y)
LINE_END = (TARGET_WIDTH, LINE_Y)

# Mapping class IDs ke nama
CLASS_NAMES = {
    2: 'Car',
    3: 'Motorcycle',
    5: 'Bus',
    7: 'Truck'
}

# --- Variabel Global untuk Analytics & FPS ---
analytics_count = {name: 0 for name in CLASS_NAMES.values()}
tracked_vehicles_ids = set() 
current_fps = 0.0 

# --- Inisialisasi Model & Perangkat ---
# Pengecekan ketersediaan GPU
if torch.cuda.is_available():
    DEVICE = '0' # Menggunakan GPU pertama
    USE_HALF = True # Menggunakan presisi setengah (FP16) untuk kecepatan
    print("✅ GPU CUDA terdeteksi. Inferensi akan menggunakan GPU.")
else:
    DEVICE = 'cpu'
    USE_HALF = False
    print("❌ GPU CUDA tidak terdeteksi. Inferensi akan menggunakan CPU.")

model = YOLO(MODEL_NAME)
app = Flask(__name__)

# --- Fungsi Pengambilan Stream YouTube (480p) ---
def get_youtube_stream_url(url, quality='480p'):
    try:
        ydl_opts = {
            'format': f'bestvideo[height<=?{quality[:-1]}]+bestaudio/best',
            'quiet': True,
            'skip_download': True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=False)
            stream_url = info_dict.get('url', info_dict.get('formats')[0]['url'])
            return stream_url
    except Exception:
        return None

# --- Fungsi Deteksi, Tracking, Counting, dan Klasifikasi ---

def generate_frames():
    global analytics_count, tracked_vehicles_ids, current_fps
    
    stream_url = get_youtube_stream_url(YOUTUBE_URL) or YOUTUBE_URL
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        print("FATAL ERROR: Tidak dapat membuka stream video.")
        return

    prev_time = time.time()
    
    while True:
        start_time = time.time() 
        
        ret, frame = cap.read()
        if not ret:
            cap.release()
            cap = cv2.VideoCapture(stream_url) 
            time.sleep(1)
            continue
        
        frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))

        # 1. Jalankan Tracking dengan Optimasi GPU (device & half)
        results = model.track(frame, 
                              persist=True, 
                              classes=VEHICLE_CLASSES, 
                              verbose=False,
                              imgsz=TARGET_WIDTH,
                              device=DEVICE, 
                              half=USE_HALF) 
        
        annotated_frame = frame
        cv2.line(annotated_frame, LINE_START, LINE_END, (0, 0, 255), 2)
        
        if results and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            
            for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                x1, y1, x2, y2 = box
                
                centroid_x = (x1 + x2) // 2
                centroid_y = (y1 + y2) // 2
                
                vehicle_class = CLASS_NAMES.get(class_id, "Unknown")
                
                # Tampilan Klasifikasi
                label = vehicle_class
                cv2.putText(annotated_frame, label, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Logika Penghitungan
                if (centroid_y > LINE_Y - 10) and (centroid_y < LINE_Y + 10) and (track_id not in tracked_vehicles_ids):
                    analytics_count[vehicle_class] = analytics_count.get(vehicle_class, 0) + 1 # Pastikan inisialisasi aman
                    tracked_vehicles_ids.add(track_id)
                    cv2.line(annotated_frame, LINE_START, LINE_END, (0, 255, 255), 4) 
        
        # 2. Hitung FPS dan Simpan ke Global
        curr_time = time.time()
        current_fps = 1 / (curr_time - start_time) if (curr_time - start_time) > 0 else 0 
        
        # Tampilkan FPS di sudut atas video
        cv2.putText(annotated_frame, f"FPS: {current_fps:.2f} ({'GPU' if DEVICE != 'cpu' else 'CPU'})", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # 3. Encoding dan Streaming: KUALITAS RENDAH (40)
        ret, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 40]) 
        if not ret:
            continue

        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# --- Rute Flask ---

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# RUTE ANALITIK: Mengambil data dan FPS
@app.route('/analytics_data')
def analytics_data():
    global analytics_count, current_fps
    
    data = analytics_count.copy()
    data['Total'] = sum(analytics_count.values())
    data['FPS'] = f"{current_fps:.2f}"
    return jsonify(data)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    print(f"Menggunakan Model: {MODEL_NAME}")
    print("Akses website di http://127.0.0.1:5000/")
    # Menghapus debug=True jika di lingkungan produksi, namun dipertahankan untuk pengembangan
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)