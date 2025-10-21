import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
import supervision as sv
import os, csv, datetime

# ======================================
#           GLOBAL SETTINGS
# ======================================

ROI_TOP_RATIO = 0.35
ROI_BOTTOM_RATIO = 0.85
enter_count = 0
exit_count = 0
previous_positions = {}
counted_ids = {}

# Log file setup
LOG_PATH = "logs/footfall_log.csv"
os.makedirs("logs", exist_ok=True)


# ======================================
#       PYTORCH SAFE GLOBALS FIX
# ======================================
from ultralytics.nn import modules as yolo_modules
yolo_conv = yolo_modules.conv
yolo_block = yolo_modules.block
yolo_head = yolo_modules.head

torch.serialization.add_safe_globals([
    DetectionModel,
    yolo_conv.Conv,
    yolo_conv.Concat,
    yolo_block.C2f,
    yolo_block.Bottleneck,
    yolo_block.SPPF,
    yolo_block.DFL,
    yolo_head.Detect,
    torch.nn.modules.container.Sequential,
    torch.nn.modules.container.ModuleList,
    torch.nn.modules.conv.Conv2d,
    torch.nn.modules.batchnorm.BatchNorm2d,
    torch.nn.modules.activation.ReLU,
    torch.nn.modules.activation.SiLU,
    torch.nn.modules.linear.Linear,
    torch.nn.modules.pooling.MaxPool2d,
    torch.nn.modules.dropout.Dropout,
    torch.nn.modules.upsampling.Upsample,
])

# ======================================
#        YOLOv8 + TRACKER INIT
# ======================================
model = YOLO("yolov8n.pt")  # YOLOv8 Nano model
tracker = sv.ByteTrack()


# ======================================
#        LOGGING FUNCTION
# ======================================
def log_event(person_id, direction):
    """Append IN/OUT event with timestamp to CSV."""
    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.datetime.now(), person_id, direction])


# ======================================
#         DRAWING OVERLAYS
# ======================================
def draw_overlay(frame, roi_top, roi_bottom, enter_count, exit_count):
    overlay = frame.copy()
    h, w = frame.shape[:2]

    cv2.rectangle(overlay, (0, roi_top), (w, roi_bottom), (0, 255, 0), 2)
    cv2.putText(overlay, "ROI BAND", (10, roi_top - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.putText(overlay, f"IN: {enter_count}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 0), 2)
    cv2.putText(overlay, f"OUT: {exit_count}", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    return overlay


# ======================================
#          FRAME PROCESSING
# ======================================
def process_frame(frame, frame_idx):
    global enter_count, exit_count

    results = model(frame, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = detections[detections.class_id == 0]  # persons only

    h, w = frame.shape[:2]
    roi_top = int(h * ROI_TOP_RATIO)
    roi_bottom = int(h * ROI_BOTTOM_RATIO)
    roi_mid = int((roi_top + roi_bottom) / 2)

    annotated = frame.copy()

    if len(detections) == 0:
        return draw_overlay(annotated, roi_top, roi_bottom, enter_count, exit_count)

    tracked = tracker.update_with_detections(detections)

    if tracked.tracker_id is not None:
        for xyxy, tid in zip(tracked.xyxy, tracked.tracker_id):
            if tid is None:
                continue

            x1, y1, x2, y2 = xyxy.astype(int)
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

            prev_y = previous_positions.get(tid)
            previous_positions[tid] = cy

            if prev_y is not None:
                # --- Enter logic ---
                if prev_y < roi_mid and cy > roi_mid:
                    if counted_ids.get(tid) != "in":
                        enter_count += 1
                        counted_ids[tid] = "in"
                        log_event(tid, "IN")
                        print(f"[Frame {frame_idx}] ID {tid} ENTERED ↓")

                # --- Exit logic ---
                elif prev_y > roi_mid and cy < roi_mid:
                    if counted_ids.get(tid) != "out":
                        exit_count += 1
                        counted_ids[tid] = "out"
                        log_event(tid, "OUT")
                        print(f"[Frame {frame_idx}] ID {tid} EXITED ↑")

            # Draw visuals
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 255), 2)
            cv2.putText(annotated, f"ID:{tid}", (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.circle(annotated, (cx, cy), 4, (0, 255, 255), -1)

    annotated = draw_overlay(annotated, roi_top, roi_bottom, enter_count, exit_count)
    return annotated


# ======================================
#          WEBCAM PROCESSING
# ======================================
def process_webcam():
    cap = cv2.VideoCapture(0)
    frame_idx = 0
    print("🎥 Running webcam... Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        annotated = process_frame(frame, frame_idx)
        cv2.imshow("Footfall Counter - Webcam", annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# ======================================
#          VIDEO PROCESSING
# ======================================
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    print(f"🎬 Processing video: {video_path}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        annotated = process_frame(frame, frame_idx)
        cv2.imshow("Footfall Counter - Video", annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# ======================================
#              MAIN MENU
# ======================================
def main_menu():
    print("\nAI Footfall Counter")
    print("1️⃣  Run Webcam Stream")
    print("2️⃣  Process Video File")
    print("3️⃣  Exit")

    choice = input("Choose (1/2/3): ").strip()

    if choice == "1":
        process_webcam()
    elif choice == "2":
        path = input("Enter video path: ").strip()
        process_video(path)
    else:
        print("Exiting...")


if __name__ == "__main__":
    main_menu()
