**AI Footfall Counter using Computer Vision**

This project is a computer vision-based footfall counter that detects and counts the number of people entering and exiting a predefined region of interest (ROI) in a video stream. The goal is to provide an automated solution for monitoring crowd flow and occupancy using object detection and tracking.

Problem Statement

The task was to design an AI-based footfall counter capable of accurately identifying people and counting their movement across an ROI in a given video feed or live stream. The system should maintain separate counts for entries and exits while ensuring that repeated detections of the same person are avoided.

Approach

YOLOv8 from the Ultralytics framework was used for real-time person detection.

The Supervision library’s ByteTrack algorithm was integrated for object tracking, which assigns unique IDs to each detected person to maintain tracking consistency across frames.

A region of interest (ROI) band was defined across the frame. When a tracked person’s centroid crosses this region in a particular direction, the system updates the entry or exit count accordingly.

OpenCV was used to process frames, visualize bounding boxes, ROI lines, and live counts directly on the output stream.

The solution works for both live webcam streams and pre-recorded video files.

Implementation Details

Model: YOLOv8n pretrained on the COCO dataset.

Tracker: ByteTrack via the Supervision library.

Environment: Python with OpenCV for frame handling and display.

ROI Logic: A horizontal band defined between 35% and 85% of the frame height to detect directional crossings.

Output: Processed frames showing live detections, ROI lines, and updated entry/exit counts.

Results and Observations

The system successfully detects and tracks people in real time.

The counting mechanism updates accurately when people move across the ROI.

For scenes with strong perspective (e.g., corridor views), adjusting ROI ratios or using a trapezoidal ROI improves performance.

The application runs smoothly on standard hardware and supports both saved videos and webcam input.

Reflection

The main challenge was designing a robust counting mechanism that differentiates between people entering and exiting within a perspective view. Using a centroid-based tracking approach and an adaptive ROI band helped improve accuracy. Future enhancements could include perspective correction, dynamic ROI adjustment, and web-based visualization for real-time analytics.

**Expected OUTPUT**
<img width="1918" height="1079" alt="image" src="https://github.com/user-attachments/assets/7377e78f-454f-4339-a8b9-4f23babbdfd7" />
<img width="734" height="324" alt="image" src="https://github.com/user-attachments/assets/37bd5d62-5954-436a-a162-aaabe1458e06" />
<img width="849" height="702" alt="image" src="https://github.com/user-attachments/assets/45c5a88a-68ec-472c-a345-7b2e898a78f4" />


