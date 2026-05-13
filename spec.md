# Vehicle Proximity Analysis - System Architecture

## Objective
Develop a Python-based system that processes a traffic video to identify the exact frame and timestamp where each unique vehicle is closest to the camera (CPA - Closest Point of Approach).

## Tech Stack
- Python 3.10+
- OpenCV (`cv2`) for video I/O and visualization.
- Ultralytics YOLO (`ultralytics`) for Object Detection and Tracking (ByteTrack/BoT-SORT).

## Core Algorithm & Logic
1. **Camera Perspective:** The camera is static and positioned high above the road. Vehicles travel towards or away from the camera.
2. **Proximity Metric:** Since the camera looks down, the bottom edge of the bounding box (`y2` in `[x1, y1, x2, y2]`) serves as the proxy for distance. **Higher `y2` value = closer to the camera.**
3. **State Management:** The system must maintain a memory (dictionary) for tracked vehicles: `track_id -> {max_y, cpa_location, best_frame, best_timestamp, best_bbox}`.
4. **Update Rule:** For every frame, if a tracked vehicle's current `y2` > its stored `max_y`, update its state. On update, store the bottom-center contact point as `cpa_location = ((x1+x2)/2, y2)` — the ground-plane anchor point, more semantically precise than `y2` alone.

## Required Outputs
1. **Processed Video:** Visual aid showing bounding boxes, track IDs, and a visual indicator when a vehicle hits its CPA.
2. **Data Export:** A JSON/CSV file listing each vehicle's `track_id`, `best_frame`, `best_timestamp`, and `bbox` at the CPA.
