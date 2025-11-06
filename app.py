import os
from utils.frame_utils import extract_frames
from utils.detect_utils import load_model, detect_objects
from utils.file_utils import create_output_folder

# Paths
VIDEO_PATH = "data/videos/cars1.mp4"
FRAME_DIR = "data/frames"
OUTPUT_BASE = "data/output"
MODEL_PATH = "models/yolov5s.pt"

if __name__ == "__main__":
    print("🚀 Memulai deteksi mobil...")

    # 1️⃣ Ekstrak frame dari video
    print("🎞️  Mengekstrak frame...")
    total_frames = extract_frames(VIDEO_PATH, FRAME_DIR, frame_skip=15)
    print(f"✅ {total_frames} frame tersimpan di {FRAME_DIR}")

    # 2️⃣ Load model YOLO
    print("🧠 Memuat model...")
    model = load_model(MODEL_PATH)
    print("✅ Model berhasil dimuat")
    
    # 3️⃣ Buat folder output baru
    output_folder = create_output_folder(OUTPUT_BASE)
    print(f"📁 Folder output: {output_folder}")

    # 4️⃣ Jalankan deteksi
    print("🔍 Melakukan deteksi...")
    detect_objects(model, FRAME_DIR, output_folder)
    print(f"✅ Deteksi selesai! Hasil tersimpan di {output_folder}")
