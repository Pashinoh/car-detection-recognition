import os
import io
import csv
import time
import cv2
import yt_dlp
from ultralytics import YOLO
from flask import Flask, Response, render_template, request, redirect, url_for, jsonify, make_response
import torch

# --- Konfigurasi Dasar ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

MODEL_NAME = 'yolov8n.pt'
VEHICLE_CLASSES = [2, 3, 5, 7]
CLASS_NAMES = {2: 'Car', 3: 'Motorcycle', 5: 'Bus', 7: 'Truck'}

TARGET_WIDTH = 640
TARGET_HEIGHT = 360
LINE_Y = int(TARGET_HEIGHT * 0.6)
LINE_START = (0, LINE_Y)
LINE_END = (TARGET_WIDTH, LINE_Y)

# --- Global state untuk analytics / log ---
analytics_count = {name: 0 for name in CLASS_NAMES.values()}
tracked_vehicles_ids = set()
detection_log = []  # list of dicts: {timestamp, elapsed_s, class, track_id, frame_no}
current_fps = 0.0
CURRENT_STREAM_URL = None
IS_STREAMING = False

# --- GPU/CPU ---
if torch.cuda.is_available():
    DEVICE = '0'
    USE_HALF = True
    print("✅ GPU CUDA terdeteksi. Menggunakan GPU.")
else:
    DEVICE = 'cpu'
    USE_HALF = False
    print("❌ GPU tidak terdeteksi. Menggunakan CPU.")

# Inisialisasi model (akan memuat model ke memori)
model = YOLO(MODEL_NAME)

# --- Helper ambil stream YouTube (jika link YouTube diberikan) ---
def get_youtube_stream_url(url, quality='480p'):
    try:
        ydl_opts = {
            'format': f'bestvideo[height<=?{quality[:-1]}]+bestaudio/best',
            'quiet': True,
            'skip_download': True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=False)
            # pilih url stream
            if 'url' in info_dict:
                return info_dict['url']
            formats = info_dict.get('formats')
            if formats:
                return formats[-1]['url']
            return None
    except Exception as e:
        print("get_youtube_stream_url error:", e)
        return None

# --- Generator frame untuk streaming dan deteksi ---
def generate_frames():
    global analytics_count, tracked_vehicles_ids, current_fps, CURRENT_STREAM_URL, IS_STREAMING, detection_log

    if CURRENT_STREAM_URL is None:
        print("⚠️ Tidak ada video aktif.")
        return

    # reset tracking counters & log ketika mulai video baru
    analytics_count = {name: 0 for name in CLASS_NAMES.values()}
    tracked_vehicles_ids = set()
    detection_log = []

    # Resolusi / sumber stream
    stream_url = get_youtube_stream_url(CURRENT_STREAM_URL) or CURRENT_STREAM_URL
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        print("❌ Gagal membuka video:", stream_url)
        IS_STREAMING = False
        return

    IS_STREAMING = True
    print(f"🎥 Memulai analisis pada: {CURRENT_STREAM_URL}")
    start_time = time.time()
    frame_no = 0

    while True:
        start_frame_time = time.time()
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Frame tidak tersedia / video selesai.")
            break

        frame_no += 1
        # jaga ukuran agar sesuai target
        try:
            frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))
        except Exception:
            # jika frame corrupt, skip
            continue

        # jalankan tracking/deteksi
        results = model.track(frame,
                              persist=True,
                              classes=VEHICLE_CLASSES,
                              verbose=False,
                              imgsz=TARGET_WIDTH,
                              device=DEVICE,
                              half=USE_HALF)

        annotated_frame = frame.copy()
        cv2.line(annotated_frame, LINE_START, LINE_END, (0, 0, 255), 2)

        if results and len(results) > 0 and getattr(results[0].boxes, 'id', None) is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)

            for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                x1, y1, x2, y2 = box
                centroid_y = (y1 + y2) // 2
                vehicle_class = CLASS_NAMES.get(int(class_id), 'Unknown')

                # gambar kotak & label
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_frame, vehicle_class, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                # logic crossing garis (hanya hitung ketika centroid melewati garis dan id belum tercatat)
                if (LINE_Y - 10 < centroid_y < LINE_Y + 10) and (int(track_id) not in tracked_vehicles_ids):
                    analytics_count[vehicle_class] = analytics_count.get(vehicle_class, 0) + 1
                    tracked_vehicles_ids.add(int(track_id))
                    cv2.line(annotated_frame, LINE_START, LINE_END, (0, 255, 255), 3)

                    # catat detail rekapan: waktu relatif, timestamp, class, track_id, frame
                    elapsed = time.time() - start_time
                    ts = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
                    detection_log.append({
                        'timestamp': ts,
                        'elapsed_s': round(elapsed, 2),
                        'class': vehicle_class,
                        'track_id': int(track_id),
                        'frame_no': frame_no
                    })

        # hitung FPS
        now = time.time()
        current_fps = 1.0 / (now - start_frame_time) if (now - start_frame_time) > 0 else 0.0
        cv2.putText(annotated_frame, f"FPS: {current_fps:.2f} ({'GPU' if DEVICE != 'cpu' else 'CPU'})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # encode dan kirim frame
        ret2, buffer = cv2.imencode('.jpg', annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
        if not ret2:
            continue
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()
    IS_STREAMING = False
    print("ℹ️ Streaming/deteksi dihentikan.")

# --- Routes Flask ---
@app.route('/')
def index():
    return render_template('index.html')  # styling asli dipertahankan di template

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/analytics_data')
def analytics_data():
    # kembalikan counts + total + FPS
    data = analytics_count.copy()
    data['Total'] = sum(analytics_count.values())
    data['FPS'] = f"{current_fps:.2f}"
    return jsonify(data)

@app.route('/detection_log')
def get_detection_log():
    # kembalikan seluruh log deteksi (list)
    return jsonify(detection_log)

@app.route('/set_video', methods=['POST'])
def set_video():
    global CURRENT_STREAM_URL

    # reset analytics & log saat user memilih video baru
    # (generate_frames juga me-reset, tetapi kita reset lebih awal untuk UI)
    # NOTE: actual reset and re-init dilakukan di generate_frames ketika mulai streaming
    video_url = request.form.get('video_url', '').strip()
    uploaded_file = request.files.get('video_file')

    if video_url:
        CURRENT_STREAM_URL = video_url
        print("🔗 URL video diterima:", CURRENT_STREAM_URL)
    elif uploaded_file and uploaded_file.filename != '':
        filename = uploaded_file.filename
        safe_name = filename.replace(' ', '_')
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], safe_name)
        uploaded_file.save(save_path)
        CURRENT_STREAM_URL = save_path
        print("📁 File diunggah:", save_path)
    else:
        # tidak ada input: kembali ke index
        return redirect(url_for('index'))

    # redirect kembali ke halaman utama — streaming akan mulai ketika /video_feed diakses
    return redirect(url_for('index'))

@app.route('/download_csv')
def download_csv():
    # buat CSV dari detection_log di memory
    global detection_log
    si = io.StringIO()
    writer = csv.writer(si)
    # header
    writer.writerow(['timestamp', 'elapsed_s', 'class', 'track_id', 'frame_no'])
    for entry in detection_log:
        writer.writerow([entry.get('timestamp'), entry.get('elapsed_s'), entry.get('class'),
                         entry.get('track_id'), entry.get('frame_no')])

    output = make_response(si.getvalue())
    output.headers["Content-Disposition"] = "attachment; filename=detection_log.csv"
    output.headers["Content-Type"] = "text/csv; charset=utf-8"
    return output

# --- Run app ---
if __name__ == '__main__':
    print(f"Menggunakan Model: {MODEL_NAME}")
    print("Buka http://127.0.0.1:5000/")
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
