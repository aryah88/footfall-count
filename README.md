**AI-Based Footfall Counter using Computer Vision**

This project implements an AI-driven system that automatically counts the number of people entering and exiting a region of interest (ROI) in real-time video streams or recorded footage. Each detected person is assigned a unique ID through object tracking, which allows the system to follow individuals across frames and avoid duplicate detections.

The counter logic works by comparing a tracked person’s position across consecutive frames relative to a defined ROI band. When a person’s trajectory crosses the upper and lower boundaries of the ROI in a downward direction, it is recorded as an entry; when the movement is in the opposite direction, it is registered as an exit. Each ID is uniquely associated with a detected person during their visible presence in the frame, ensuring consistent tracking and accurate footfall monitoring.

A future improvement involves refining the logic to make the counting process strictly singular per unique ID, ensuring that one person is counted only once during their full path across the ROI even under prolonged visibility or occlusions.

The entire system uses YOLOv8 for human detection and ByteTrack for identity-preserving object tracking. The detection results are processed in real time using OpenCV, with tracking data stored in structured logs. These logs feed a FastAPI-based backend that provides analytical endpoints and visual insights.

The FastAPI layer generates JSON-based statistics for total entries, exits, and current occupancy levels. It also produces automatic charts and trends using Matplotlib and Seaborn to visualize cumulative activity, entry-exit distributions, and overall flow patterns. The charts update dynamically and can be accessed through lightweight API calls, offering quick visibility into real-time and historical footfall data without any external frontend framework.

This combination of real-time computer vision, tracking consistency, and automated analytics provides a practical approach to people counting and flow analysis, applicable to environments such as retail spaces, campuses, and smart surveillance systems.

**Expected OUTPUT**
<img width="1918" height="1079" alt="image" src="https://github.com/user-attachments/assets/7377e78f-454f-4339-a8b9-4f23babbdfd7" />
<img width="734" height="324" alt="image" src="https://github.com/user-attachments/assets/37bd5d62-5954-436a-a162-aaabe1458e06" />
<img width="849" height="702" alt="image" src="https://github.com/user-attachments/assets/45c5a88a-68ec-472c-a345-7b2e898a78f4" />


