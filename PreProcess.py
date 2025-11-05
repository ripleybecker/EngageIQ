from ultralytics import YOLO
import cv2
import pandas as pd

# --- CONFIG ---
VIDEO_PATH = "IMG_5168.MOV"
OUTPUT_CSV = "detections.csv"
MODEL_PATH = "yolo11n.pt"  # fast but accurate

# --- INIT ---
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_id = 0
records = []

# Define zones (x1, y1, x2, y2)
zones = [
    (200, 200, 600, 700),  # Exhibit Zone 1
    (900, 200, 1200, 700)   # Exhibit Zone 2
]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)
    detections = results[0].boxes

    for box in detections:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

        # only track people
        if model.names[cls] == 'person':
            records.append({
                "frame": frame_id,
                "time_s": frame_id / fps,
                "x": cx,
                "y": cy,
                "conf": conf,
                "zone": next((i+1 for i, (zx1, zy1, zx2, zy2) in enumerate(zones)
                              if zx1 < cx < zx2 and zy1 < cy < zy2), 0)
            })

    frame_id += 1

cap.release()
pd.DataFrame(records).to_csv(OUTPUT_CSV, index=False)
print(f"✅ Detection preprocessing complete. Saved {len(records)} detections to {OUTPUT_CSV}")
