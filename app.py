import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv

# === GLOBAL SETTINGS ===
ROI_TOP_RATIO = 0.35
ROI_BOTTOM_RATIO = 0.85
enter_count = 0
exit_count = 0
previous_positions = {}

# === Initialize YOLOv8 and Tracker ===
model = YOLO("yolov8n.pt")
tracker = sv.ByteTrack()

# === Draw Overlay ===
def draw_overlay(frame, roi_top, roi_bottom, enter, exit):
    overlay = frame.copy()
    cv2.line(overlay, (0, roi_top), (frame.shape[1], roi_top), (0, 255, 0), 2)
    cv2.line(overlay, (0, roi_bottom), (frame.shape[1], roi_bottom), (0, 255, 0), 2)
    cv2.rectangle(overlay, (10, 10), (220, 80), (0, 0, 0), -1)
    cv2.putText(overlay, f"IN: {enter}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.putText(overlay, f"OUT: {exit}", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    return overlay


# === Frame Processor ===
def process_frame(frame, frame_idx):
    global enter_count, exit_count, previous_positions

    results = model(frame, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = detections[detections.class_id == 0]

    h, w = frame.shape[:2]
    roi_top = int(h * ROI_TOP_RATIO)
    roi_bottom = int(h * ROI_BOTTOM_RATIO)
    annotated = frame.copy()

    if len(detections) == 0:
        return draw_overlay(annotated, roi_top, roi_bottom, enter_count, exit_count)

    tracked = tracker.update_with_detections(detections)

    if tracked.tracker_id is None:
        return draw_overlay(annotated, roi_top, roi_bottom, enter_count, exit_count)

    for xyxy, tid in zip(tracked.xyxy, tracked.tracker_id):
        if tid is None:
            continue

        x1, y1, x2, y2 = xyxy.astype(int)
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

        prev_y = previous_positions.get(tid, None)
        previous_positions[tid] = cy

        if prev_y is not None:
            if prev_y < roi_bottom <= cy:
                enter_count += 1
                print(f"[{frame_idx}] ID {tid} ENTERED ↓")
            elif prev_y > roi_top >= cy:
                exit_count += 1
                print(f"[{frame_idx}] ID {tid} EXITED ↑")

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 255), 2)
        cv2.putText(annotated, f"ID:{tid}", (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.circle(annotated, (cx, cy), 4, (0, 255, 255), -1)

    annotated = draw_overlay(annotated, roi_top, roi_bottom, enter_count, exit_count)
    return annotated


# === Process Video File (LIVE Preview Added) ===
def process_video(video_path):
    global enter_count, exit_count, previous_positions
    enter_count, exit_count = 0, 0
    previous_positions = {}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Could not open video file.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path = "processed_output.mp4"
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    frame_idx = 0
    print("🎬 Processing video... Press 'q' to quit early.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        annotated = process_frame(frame, frame_idx)
        out.write(annotated)

        # --- LIVE PREVIEW ---
        cv2.imshow("🧠 Processing Video", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("🛑 User stopped processing.")
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"✅ Done! Saved processed video to {out_path}")


# === Webcam Stream ===
def process_webcam():
    global enter_count, exit_count, previous_positions
    enter_count, exit_count = 0, 0
    previous_positions = {}

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not access webcam.")
        return

    frame_idx = 0
    print("🎥 Running webcam. Press 'q' to stop.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        annotated = process_frame(frame, frame_idx)
        cv2.imshow("🧠 Footfall Counter (Webcam)", annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("🛑 Webcam stream ended.")


# === Main Menu ===
def main_menu():
    print("\n🧠 AI Footfall Counter — ROI Band Mode")
    print("1️⃣  Run Webcam Stream")
    print("2️⃣  Process Video File")
    choice = input("Choose (1/2): ").strip()

    if choice == "1":
        process_webcam()
    elif choice == "2":
        path = input("Enter video file path: ").strip()
        process_video(path)
    else:
        print("❌ Invalid choice.")


if __name__ == "__main__":
    main_menu()
