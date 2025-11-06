import cv2
import os

def extract_frames(video_path, output_folder, frame_skip=15):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_skip == 0:
            resized = cv2.resize(frame, (640, 360))
            frame_name = f"frame_{saved_count:04d}.jpg"
            cv2.imwrite(os.path.join(output_folder, frame_name), resized)
            saved_count += 1

        frame_count += 1

    cap.release()
    return saved_count
