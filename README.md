# Vehicle detection

Detects the **Closest Point of Approach (CPA)** for each vehicle in a traffic video — the exact frame and timestamp where every tracked vehicle was nearest to the camera.


[![Demo](demo_thumbnail.jpg)](clip_1_result_.mp4)


## How It Works

The camera is mounted high above the road looking down. In this top-down perspective, a vehicle's vertical position in the frame directly correlates with its physical distance from the camera:

- **Higher `y2` (bottom edge of bounding box) = closer to the camera**

For every frame, the system compares each vehicle's current `y2` against its stored maximum. When `y2` exceeds the record, the record is updated. By the end of the video, each vehicle ID has a single authoritative CPA entry.

There is a deliberate distinction between two roles:
- **Trigger (`y2` only):** The `if` comparison uses only the vertical value — `x` encodes lane position, not depth, so it plays no part in deciding closeness.
- **Record (`cpa_location`):** Once a new maximum is confirmed, the system stores the bottom-center contact point `((x1+x2)/2, y2)` as `cpa_location` — the point where the vehicle's wheels meet the road plane.

```
track_id → { max_y, cpa_location, best_frame, best_timestamp, best_bbox }
```

## Project Structure

```
vehicle_detection/
├── main.py          # Full implementation (VideoProcessor + ProximityTracker + BBoxSmoother)
├── bytetrack.yaml   # ByteTrack tracker configuration
├── spec.md          # Architecture specification
├── report.md        # Technical report (logic, accuracy, limitations, next steps)
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py <input_video> [--output output.mp4] [--model yolov8s.pt] [--scale 0.5]
```

**Example:**
```bash
python main.py traffic.mp4 --output result.mp4
```

## Outputs

| File | Description |
|---|---|
| `result.mp4` | Annotated video with bounding boxes and event log |
| `result_cpa.json` | Per-vehicle CPA data (track ID, frame, timestamp, cpa_location, bbox) |
| `result_cpa.csv` | Same data in CSV format |

### Video Annotations

- **Grey box** — vehicle being tracked
- **Green box** — vehicle has crossed the proximity threshold (bottom 12% of frame)
- **Top-right panel** — confirmed vehicle passes with fade-in timestamps
- **Top-left HUD** — current frame index and timestamp

## Output Data Format

```json
[
  {
    "track_id": 3,
    "class": "vehicle",
    "max_y": 641.2,
    "cpa_location": [505.25, 641.2],
    "best_frame": 187,
    "best_timestamp": 6.2333,
    "best_bbox": [412.1, 501.8, 598.4, 641.2]
  }
]
```

`cpa_location` is `[cx, y2]` where `cx = (x1+x2)/2` — the bottom-center of the bounding box, representing the vehicle's ground contact point.

## Tech Stack

- **Python 3.10+**
- **[Ultralytics YOLO](https://github.com/ultralytics/ultralytics)** (`yolov8s`) — object detection and multi-object tracking (ByteTrack)
- **OpenCV** — video I/O and visualization

## Limitations

- Occlusions may cause a vehicle to re-appear with a new track ID, splitting its record
- The `y2` proxy assumes a near-nadir camera angle — results degrade if the camera is significantly tilted
