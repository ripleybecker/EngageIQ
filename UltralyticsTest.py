import cv2
import time
from ultralytics import YOLO
import numpy as np
import json

# Load YOLO model
model = YOLO("yolo11n.pt")  # You can swap for yolo11s.pt if you want more accuracy

# Open webcam
cap = cv2.VideoCapture(0)

# --- Define two exhibit zones (x1, y1, x2, y2) ---
ZONE_1 = (200, 200, 600, 700)  # Left zone
ZONE_2 = (900, 200, 1200, 700)  # Right zone

# --- Tracking data structures ---
people_data = {}           # {id: {"zone": int, "start": time.time()}}
completed_durations = {1: [], 2: []}  # Store past durations per zone
next_id = 0                # ID counter for new visitors

def get_center(box):
    """Compute the center of a bounding box."""
    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

def in_zone(center, zone):
    """Check if a point (center) is within a rectangular zone."""
    x, y = center
    x1, y1, x2, y2 = zone
    return x1 <= x <= x2 and y1 <= y <= y2

def average(lst):
    """Compute the average of a list safely."""
    return round(sum(lst) / len(lst), 1) if lst else 0.0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO inference
    results = model(frame, verbose=False)
    detections = results[0].boxes.data.cpu().numpy()

    current_ids_in_zone = set()  # People currently detected inside any zone

    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        if int(cls) != 0:  # Only track 'person'
            continue

        center = get_center((x1, y1, x2, y2))

        # Check if person is in zone 1 or 2
        zone_id = None
        if in_zone(center, ZONE_1):
            zone_id = 1
        elif in_zone(center, ZONE_2):
            zone_id = 2

        if zone_id:
            # Try to match with an existing tracked person
            matched_id = None
            for pid, pdata in people_data.items():
                px, py = pdata.get("last_center", (0, 0))
                if np.linalg.norm(np.array(center) - np.array((px, py))) < 50:
                    matched_id = pid
                    break

            if matched_id is None:
                # New visitor enters zone
                next_id += 1
                matched_id = next_id
                people_data[matched_id] = {"zone": zone_id, "start": time.time()}

            # Update position
            people_data[matched_id]["last_center"] = center
            current_ids_in_zone.add(matched_id)

            # Draw bounding box and ID
            color = (0, 255, 0) if zone_id == 1 else (0, 200, 255)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, f"ID {matched_id}", (int(x1), int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)

    # Check for people who left zones
    for pid in list(people_data.keys()):
        if pid not in current_ids_in_zone:
            pdata = people_data.pop(pid)
            duration = round(time.time() - pdata["start"], 1)
            zone_id = pdata["zone"]
            completed_durations[zone_id].append(duration)

    with open("data.json", "w") as f:
        json.dump(completed_durations, f)

    # --- Draw exhibit zones ---
    cv2.rectangle(frame, (ZONE_1[0], ZONE_1[1]), (ZONE_1[2], ZONE_1[3]), (255, 0, 0), 2)
    cv2.putText(frame, "Zone 1", (ZONE_1[0], ZONE_1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.rectangle(frame, (ZONE_2[0], ZONE_2[1]), (ZONE_2[2], ZONE_2[3]), (0, 165, 255), 2)
    cv2.putText(frame, "Zone 2", (ZONE_2[0], ZONE_2[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

    # --- Build dashboard window ---
    dashboard = np.zeros((400, 500, 3), dtype=np.uint8)

    # Zone 1 stats
    avg1 = average(completed_durations[1])
    cv2.putText(dashboard, "Zone 1 Stats", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(dashboard, f"Average stay: {avg1}s", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 255), 2)

    y_pos = 130
    for i, dur in enumerate(completed_durations[1][-5:][::-1]):
        cv2.putText(dashboard, f"Stay {len(completed_durations[1]) - i}: {dur}s", (30, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 255), 2)
        y_pos += 25

    # Zone 2 stats
    avg2 = average(completed_durations[2])
    cv2.putText(dashboard, "Zone 2 Stats", (280, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(dashboard, f"Average stay: {avg2}s", (280, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 150), 2)

    y_pos = 130
    for i, dur in enumerate(completed_durations[2][-5:][::-1]):
        cv2.putText(dashboard, f"Stay {len(completed_durations[2]) - i}: {dur}s", (280, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 150), 2)
        y_pos += 25

    # Show both windows
    cv2.imshow("Exhibit Tracking", frame)
    cv2.imshow("Dashboard", dashboard)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()

