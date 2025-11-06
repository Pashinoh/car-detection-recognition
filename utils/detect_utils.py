from ultralytics import YOLO
import cv2
import os

def load_model(model_path):
    """Memuat model YOLO"""
    model = YOLO(model_path)
    return model

def detect_objects(model, frame_dir, output_dir):
    """Deteksi objek di setiap frame dan simpan hasilnya"""
    os.makedirs(output_dir, exist_ok=True)

    for file_name in os.listdir(frame_dir):
        frame_path = os.path.join(frame_dir, file_name)
        img = cv2.imread(frame_path)
        if img is None:
            continue

        # Ubah ukuran seragam
        img_resized = cv2.resize(img, (640, 360))

        # Deteksi objek
        results = model(img_resized, verbose=False)
        boxes = results[0].boxes
        labels = [results[0].names[int(c)] for c in boxes.cls] if boxes.cls.numel() > 0 else []

        # Simpan gambar
        if len(labels) > 0:
            # Ada deteksi → simpan gambar dengan kotak
            annotated_img = results[0].plot()
            cv2.imwrite(os.path.join(output_dir, file_name), annotated_img)
            print(f"🟢 {file_name}: {', '.join(labels)}")
        else:
            # Tidak ada deteksi → simpan gambar biasa
            cv2.imwrite(os.path.join(output_dir, file_name), img_resized)
            print(f"⚪ {file_name}: tidak ada objek terdeteksi")
