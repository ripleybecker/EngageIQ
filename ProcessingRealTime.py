"""
Video zone-dwell tracker (uses video frame indices for timing).
Requires: pip install ultralytics opencv-python numpy
Replace MODEL_PATH and VIDEO_PATH as needed.
"""

import cv2
import numpy as np
from ultralytics import YOLO

# ----------------- CONFIG -----------------
MODEL_PATH = "yolo11n.pt"             # one of: yolov8n.pt / yolo11n.pt / yolo11n-seg.pt
VIDEO_PATH = "IMG_5168.MOV"           # replace with your video path
DIST_THRESHOLD = 60                   # px for centroid matching across frames 
MIN_STAY_SECONDS = 0.5                # ignore very short blips
SHOW_GUI = True                       # set False to run headless
# Define zones as (x1,y1,x2,y2). Example: two zones
ZONES = {
    1: (200, 200, 600, 700),
    2: (900, 200, 1200, 700),
}
# ------------------------------------------

# Load model
model = YOLO(MODEL_PATH)

# Open video
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise SystemExit(f"Cannot open video: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
MAX_MISSED_FRAMES = int(fps * 1.0)  # roughly 1.0 seconds of forgiveness

frame_idx = 0

# Tracking data structures
next_id = 1
# active_tracks: id -> { 'zone': zone_id, 'entry_frame': int, 'last_frame': int, 'last_center': (x,y), 'missed': int }
active_tracks = {}
# completed durations per zone (seconds)
completed = {z: [] for z in ZONES.keys()}

def center_of_box(box):
    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

def point_in_zone(pt, zone):
    x, y = pt
    x1, y1, x2, y2 = zone
    return x1 <= x <= x2 and y1 <= y <= y2

def euclid(a, b):
    return ((a[0]-b[0])**2 + (a[1]-b[1])**2)**0.5

# Helper: draw debugging overlays
def draw_zones(img):
    for zid, z in ZONES.items():
        x1,y1,x2,y2 = z
        color = (0,200,255) if zid==2 else (255,0,0)
        cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
        cv2.putText(img, f"Zone {zid}", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

# MAIN LOOP
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1

    # Run YOLO on this frame (people only)
    results = model(frame, verbose=False)
    # results[0].boxes.data -> each row: [x1,y1,x2,y2, conf, cls]
    dets = results[0].boxes.data.cpu().numpy() if len(results[0].boxes) > 0 else np.empty((0,6))

    # Build detection list: only person class (COCO class 0)
    detections = []  # list of dicts: {'bbox':(x1,y1,x2,y2), 'center':(cx,cy)}
    for row in dets:
        x1,y1,x2,y2,conf,cls = row
        if int(cls) != 0:
            continue
        bbox = (int(x1), int(y1), int(x2), int(y2))
        center = center_of_box(bbox)
        # determine which zone (if any)
        zone_id = 0
        for zid, z in ZONES.items():
            if point_in_zone(center, z):
                zone_id = zid
                break
        if zone_id == 0:
            # ignore people outside all zones for timing/ID creation
            continue
        detections.append({'bbox': bbox, 'center': center, 'zone': zone_id})

    # Matching detections to existing active_tracks
    matched_tracks = set()
    used_dets = set()

    # Greedy matching: for each detection, find nearest active track in same zone
    for det_idx, det in enumerate(detections):
        best_id = None
        best_dist = DIST_THRESHOLD + 1
        for tid, track in active_tracks.items():
            if track['zone'] != det['zone']:
                continue
            dist = euclid(det['center'], track['last_center'])
            if dist < best_dist:
                best_dist = dist
                best_id = tid
        if best_id is not None and best_dist <= DIST_THRESHOLD:
            # match
            active_tracks[best_id]['last_center'] = det['center']
            active_tracks[best_id]['last_frame'] = frame_idx
            active_tracks[best_id]['missed'] = 0
            matched_tracks.add(best_id)
            used_dets.add(det_idx)

    # Unmatched detections -> create new tracks (new visits)
    for det_idx, det in enumerate(detections):
        if det_idx in used_dets:
            continue
        # create new track ID
        tid = next_id
        next_id += 1
        active_tracks[tid] = {
            'zone': det['zone'],
            'entry_frame': frame_idx,
            'last_frame': frame_idx,
            'last_center': det['center'],
            'missed': 0
        }
        matched_tracks.add(tid)

    # For active tracks not matched this frame, increment missed count
    for tid, track in list(active_tracks.items()):
        if tid not in matched_tracks:
            track['missed'] += 1
            # If missed too many consecutive frames -> consider left
            if track['missed'] > MAX_MISSED_FRAMES:
                # compute duration using frame indices -> convert to seconds using fps
                entry_f = track.get('entry_frame', track.get('last_frame', frame_idx))
                exit_f = track.get('last_frame', entry_f)
                duration_s = (exit_f - entry_f + 1) / fps
                if duration_s >= MIN_STAY_SECONDS:
                    completed[track['zone']].append(round(duration_s, 2))
                del active_tracks[tid]

    # DRAW & DISPLAY
    if SHOW_GUI:
        vis = frame.copy()
        # draw all current active track labels and current detections
        for tid, track in active_tracks.items():
            cx, cy = track['last_center']
            zid = track['zone']
            color = (0,255,0) if zid==1 else (0,200,255)
            cv2.circle(vis, (int(cx), int(cy)), 6, color, -1)
            # live elapsed using frames -> seconds
            elapsed = (track['last_frame'] - track['entry_frame'] + 1) / fps
            cv2.putText(vis, f"ID {tid} {elapsed:.1f}s", (int(cx)+8, int(cy)-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        # draw zone boxes
        draw_zones(vis)
        # draw det bounding boxes for this frame
        for det in detections:
            x1,y1,x2,y2 = det['bbox']
            zid = det['zone']
            color = (0,255,0) if zid==1 else (0,200,255)
            cv2.rectangle(vis, (x1,y1),(x2,y2), color, 2)

        # dashboard: show completed durations per zone and averages
        dash = np.zeros((400, 400, 3), dtype=np.uint8)
        y = 30
        for zid in sorted(ZONES.keys()):
            avg = round(sum(completed[zid]) / len(completed[zid]), 2) if completed[zid] else 0.0
            cv2.putText(dash, f"Zone {zid} Avg: {avg}s", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
            y += 25
            cv2.putText(dash, f"Last stays:", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,150,150), 1)
            y += 20
            # show last 8 entries
            for d in completed[zid][-8:][::-1]:
                cv2.putText(dash, f"{d:.2f}s", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,180,180), 1)
                y += 18
            y += 15

        cv2.imshow("Video (annotated)", vis)
        cv2.imshow("Dashboard", dash)

        if cv2.waitKey(1) & 0xFF == 27:
            break

# At end of video, finalize remaining active tracks as left
for tid, track in list(active_tracks.items()):
    entry_f = track.get('entry_frame', track.get('last_frame'))
    exit_f = track.get('last_frame', entry_f)
    duration_s = (exit_f - entry_f + 1) / fps
    if duration_s >= MIN_STAY_SECONDS:
        completed[track['zone']].append(round(duration_s, 2))
    del active_tracks[tid]

print("Completed durations per zone:")
for zid, arr in completed.items():
    print(f"Zone {zid}: count={len(arr)} avg={round(sum(arr)/len(arr),2) if arr else 0.0} last10={arr[-10:]}")

cap.release()
if SHOW_GUI:
    cv2.destroyAllWindows()
