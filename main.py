"""
Vehicle Proximity Analysis
Identifies the Closest Point of Approach (CPA) for each tracked vehicle in a traffic video.

UI Design: Event-driven display.
  - Bounding boxes shown for all detected vehicles from the first frame they appear.
  - Box is grey by default; turns green when the vehicle reaches the epsilon line (bottom 12%).
  - CPA confirmed only when vehicle disappears AND had max_y >= epsilon line.
  - Grace period (Track Patience): vehicle must be absent for TRACK_PATIENCE consecutive
    frames before CPA is confirmed. During grace period, last known box is held ("coasting").
  - Top-right panel logs confirmed passes with vehicle class + ID and timestamp, with fade-in.
"""

import cv2
import json
import csv
import argparse
from collections import deque
from pathlib import Path
from ultralytics import YOLO

# COCO class IDs for vehicles
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# Colors (BGR)
COLOR_ROI_BOX    = (180, 180, 180)
COLOR_HUD        = (255, 255, 255)

EPSILON_THRESHOLD = 0.88   # bottom 12% of frame
TICKER_MAX        = 8
FADE_IN_FRAMES    = 20
TRACK_PATIENCE    = 30     # frames a vehicle may be absent before CPA is confirmed


class BBoxSmoother:
    """
    Separates position (cx, cy) and size (w, h) smoothing.
    Position alpha is adaptive (velocity-weighted); size alpha is fixed — prevents
    YOLO edge noise on large nearby objects from causing visible box jumping.
    """

    def __init__(self, base_alpha: float = 0.15, max_alpha: float = 0.9,
                 speed_divisor: float = 30.0, size_alpha: float = 0.35):
        self.base_alpha    = base_alpha
        self.max_alpha     = max_alpha
        self.speed_divisor = speed_divisor
        self.size_alpha    = size_alpha
        self._state:    dict[int, list[float]] = {}  # [cx, cy, w, h]
        self._velocity: dict[int, list[float]] = {}  # [vcx, vcy]

    def smooth(self, track_id: int, bbox: list[float]) -> list[float]:
        x1, y1, x2, y2 = bbox
        cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1

        if track_id not in self._state:
            self._state[track_id]    = [cx, cy, w, h]
            self._velocity[track_id] = [0.0, 0.0]
            return bbox[:]

        s_cx, s_cy, s_w, s_h = self._state[track_id]

        displacement = ((cx - s_cx) ** 2 + (cy - s_cy) ** 2) ** 0.5
        pos_alpha = min(self.max_alpha, self.base_alpha + displacement / self.speed_divisor)

        sm_cx = pos_alpha * cx + (1 - pos_alpha) * s_cx
        sm_cy = pos_alpha * cy + (1 - pos_alpha) * s_cy
        sm_w  = self.size_alpha * w + (1 - self.size_alpha) * s_w
        sm_h  = self.size_alpha * h + (1 - self.size_alpha) * s_h

        self._velocity[track_id] = [sm_cx - s_cx, sm_cy - s_cy]
        self._state[track_id]    = [sm_cx, sm_cy, sm_w, sm_h]

        return [sm_cx - sm_w / 2, sm_cy - sm_h / 2, sm_cx + sm_w / 2, sm_cy + sm_h / 2]

    def extrapolate(self, track_id: int) -> list[float]:
        """Advance position by last known velocity — call during coasting frames."""
        if track_id not in self._state:
            return []
        s_cx, s_cy, s_w, s_h = self._state[track_id]
        vcx, vcy = self._velocity[track_id]
        new_cx, new_cy = s_cx + vcx, s_cy + vcy
        self._state[track_id]    = [new_cx, new_cy, s_w, s_h]
        self._velocity[track_id] = [vcx * 0.8, vcy * 0.8]
        return [new_cx - s_w / 2, new_cy - s_h / 2, new_cx + s_w / 2, new_cy + s_h / 2]

    def cleanup(self, active_ids: set[int]) -> None:
        for tid in set(self._state.keys()) - active_ids:
            del self._state[tid]
            self._velocity.pop(tid, None)


class ProximityTracker:
    def __init__(self):
        self.records: dict[int, dict] = {}

    def update(self, track_id: int, class_name: str, bbox: list[float],
               frame_idx: int, timestamp: float) -> None:
        x1, y1, x2, y2 = bbox
        if track_id not in self.records or y2 > self.records[track_id]["max_y"]:
            cx = (x1 + x2) / 2
            self.records[track_id] = {
                "class":          class_name,
                "max_y":          y2,
                "cpa_location":   [round(cx, 2), round(y2, 2)],
                "best_frame":     frame_idx,
                "best_timestamp": round(timestamp, 4),
                "best_bbox":      [round(v, 2) for v in bbox],
            }

    def export_json(self, path: str) -> None:
        data = [{"track_id": int(tid), **info} for tid, info in self.records.items()]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[CPA] JSON saved → {path}")

    def export_csv(self, path: str) -> None:
        fieldnames = ["track_id", "class", "best_frame", "best_timestamp",
                      "max_y", "cpa_location", "best_bbox"]
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for tid, info in self.records.items():
                row = {"track_id": int(tid), **info}
                row["cpa_location"] = json.dumps(row["cpa_location"])
                row["best_bbox"]    = json.dumps(row["best_bbox"])
                writer.writerow(row)
        print(f"[CPA] CSV saved  → {path}")


class VideoProcessor:
    def __init__(self, input_path: str, output_path: str,
                 model_name: str = "yolov8n.pt", scale: float = 0.5,
                 epsilon_threshold: float = EPSILON_THRESHOLD):
        self.input_path        = input_path
        self.output_path       = output_path
        self.scale             = scale
        self.epsilon_threshold = epsilon_threshold
        self.model             = YOLO(model_name)
        self.tracker           = ProximityTracker()
        self.smoother          = BBoxSmoother()

        self._confirmed: set   = set()
        self._pass_count: int  = 0
        self._event_log: deque = deque(maxlen=TICKER_MAX)

        # Track Patience state
        self._patience:  dict[int, int]        = {}  # tid → remaining grace frames
        self._last_bbox: dict[int, list[float]] = {}  # tid → last smoothed bbox
        self._last_cls:  dict[int, str]         = {}  # tid → class name
        # Permanently latched once a vehicle crosses epsilon — stays True during coasting
        self._crossed_epsilon: set[int]         = set()

    def _open_capture(self):
        cap = cv2.VideoCapture(self.input_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {self.input_path}")
        fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return cap, fps, width, height, total

    def _make_writer(self, width: int, height: int, fps: float) -> cv2.VideoWriter:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        return cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))

    def _confirm_cpa(self, tid: int, epsilon_y_original: int) -> None:
        if tid in self._confirmed or tid not in self.tracker.records:
            return
        rec = self.tracker.records[tid]
        if rec["max_y"] < epsilon_y_original:
            return
        self._confirmed.add(tid)
        self._pass_count += 1
        self._event_log.append({
            "track_id":  tid,
            "class":     rec["class"],
            "timestamp": rec["best_timestamp"],
            "age":       0,
        })

    def _expire_track(self, tid: int, epsilon_y_original: int) -> None:
        """Fully retire a track: confirm CPA and purge all per-track state."""
        self._confirm_cpa(tid, epsilon_y_original)
        self._patience.pop(tid, None)
        self._last_bbox.pop(tid, None)
        self._last_cls.pop(tid, None)
        self._crossed_epsilon.discard(tid)

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _draw_hud(self, frame, frame_idx: int, timestamp: float) -> None:
        cv2.putText(frame, f"Frame {frame_idx}  |  {timestamp:.2f}s",
                    (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_HUD, 2)

    def _draw_epsilon_line(self, frame, epsilon_y: int) -> None:
        h, w = frame.shape[:2]
        cv2.line(frame, (0, epsilon_y), (w, epsilon_y), (0, 180, 0), 1)

    def _draw_vehicle(self, frame, bbox: list[float], track_id: int,
                      class_name: str, highlight: bool = False) -> None:
        x1, y1, x2, y2 = (int(v) for v in bbox)
        color = (0, 230, 0) if highlight else COLOR_ROI_BOX
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
        label = f"{class_name} #{track_id}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0, 0, 0), -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    def _draw_log_panel(self, frame) -> None:
        h, w    = frame.shape[:2]
        pad     = 10
        line_h  = 22
        panel_w = 260
        n       = len(self._event_log) + 1
        panel_h = pad * 2 + line_h * n
        x0      = w - panel_w - 10
        y0      = 36

        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        cv2.putText(frame, "Vehicle", (x0 + pad, y0 + pad + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 255, 180), 1)
        cv2.putText(frame, "Timestamp", (x0 + pad + 120, y0 + pad + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 255, 180), 1)
        cv2.line(frame, (x0 + pad, y0 + pad + 18),
                 (x0 + panel_w - pad, y0 + pad + 18), (80, 120, 80), 1)

        for i, event in enumerate(reversed(self._event_log)):
            alpha     = min(1.0, event["age"] / FADE_IN_FRAMES)
            y         = y0 + pad + line_h * (i + 1) + 13
            intensity = int(200 * alpha)
            color     = (intensity, 255, intensity)
            cv2.putText(frame, f"{event['class']} #{event['track_id']}",
                        (x0 + pad, y), cv2.FONT_HERSHEY_SIMPLEX, 0.44, color, 1)
            cv2.putText(frame, f"{event['timestamp']:.2f}s",
                        (x0 + pad + 120, y), cv2.FONT_HERSHEY_SIMPLEX, 0.44, color, 1)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def process(self) -> None:
        cap, fps, width, height, total = self._open_capture()

        orig_w, orig_h = width, height
        out_w = int(width  * self.scale)
        out_h = int(height * self.scale)

        epsilon_y_orig   = int(orig_h * self.epsilon_threshold)
        epsilon_y_scaled = int(out_h  * self.epsilon_threshold)

        writer = self._make_writer(out_w, out_h, fps)

        print(f"[INFO] Processing '{self.input_path}'  ({total} frames @ {fps:.1f} fps)")
        print(f"[INFO] Epsilon y={epsilon_y_scaled}  |  Patience={TRACK_PATIENCE} frames")

        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp    = frame_idx / fps
            detected_now = set()

            # Run YOLO on original frame for maximum detection stability
            results = self.model.track(frame, persist=True, verbose=False,
                                       classes=list(VEHICLE_CLASSES.keys()),
                                       iou=0.3, agnostic_nms=True,
                                       tracker="bytetrack.yaml")

            # Scale frame for drawing immediately after inference
            if self.scale != 1.0:
                frame = cv2.resize(frame, (out_w, out_h))

            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes   = results[0].boxes.xyxy.cpu().numpy()
                ids     = results[0].boxes.id.cpu().numpy().astype(int)
                classes = results[0].boxes.cls.cpu().numpy().astype(int)

                for raw_bbox, tid, cls_id in zip(boxes, ids, classes):
                    tid      = int(tid)
                    cls_name = "vehicle"
                    raw      = raw_bbox.tolist()
                    detected_now.add(tid)

                    # Reset patience — vehicle is live
                    self._patience[tid] = TRACK_PATIENCE
                    self._last_cls[tid] = cls_name

                    # CPA tracking on original-resolution coordinates
                    self.tracker.update(tid, cls_name, raw, frame_idx, timestamp)

                    # Scale → smooth → store for coasting
                    scaled_raw   = [v * self.scale for v in raw]
                    bbox_display = self.smoother.smooth(tid, scaled_raw)
                    self._last_bbox[tid] = bbox_display

                    if bbox_display[3] >= epsilon_y_scaled:
                        self._crossed_epsilon.add(tid)

                    self._draw_vehicle(frame, bbox_display, tid, cls_name,
                                       highlight=(tid in self._crossed_epsilon))

            # Handle tracks not detected this frame
            expired = []
            for tid in list(self._patience.keys()):
                if tid in detected_now:
                    continue
                self._patience[tid] -= 1
                if self._patience[tid] <= 0:
                    expired.append(tid)
                else:
                    # Coast: extrapolate position using last known velocity
                    extrapolated = self.smoother.extrapolate(tid)
                    if extrapolated:
                        x1 = max(0.0, extrapolated[0])
                        y1 = max(0.0, extrapolated[1])
                        x2 = min(float(out_w), extrapolated[2])
                        y2 = min(float(out_h), extrapolated[3])
                        if x2 > x1 and y2 > y1:
                            clamped = [x1, y1, x2, y2]
                            self._last_bbox[tid] = clamped
                            self._draw_vehicle(frame, clamped, tid,
                                               self._last_cls.get(tid, "vehicle"),
                                               highlight=(tid in self._crossed_epsilon))

            for tid in expired:
                self._expire_track(tid, epsilon_y_orig)

            # Keep smoother alive while track still has patience remaining
            self.smoother.cleanup(set(self._patience.keys()))

            for event in self._event_log:
                event["age"] += 1

            self._draw_log_panel(frame)
            self._draw_hud(frame, frame_idx, timestamp)
            writer.write(frame)

            if frame_idx % 100 == 0:
                print(f"  frame {frame_idx}/{total}  passed camera: {self._pass_count}")

            frame_idx += 1

        # Confirm all tracks still alive at end of video
        for tid in list(self._patience.keys()):
            self._expire_track(tid, epsilon_y_orig)

        cap.release()
        writer.release()
        print(f"[INFO] Output video saved → {self.output_path}")
        print(f"[INFO] Vehicles that passed the camera: {self._pass_count}")

        stem    = Path(self.output_path).stem
        out_dir = Path(self.output_path).parent
        self.tracker.export_json(str(out_dir / f"{stem}_cpa.json"))
        self.tracker.export_csv(str(out_dir / f"{stem}_cpa.csv"))


def main():
    parser = argparse.ArgumentParser(description="Vehicle Proximity Analysis (CPA detection)")
    parser.add_argument("input",      help="Path to input traffic video")
    parser.add_argument("--output",   default="output.mp4")
    parser.add_argument("--model",    default="yolov8s.pt")
    parser.add_argument("--scale",    type=float, default=0.5)
    parser.add_argument("--epsilon",  type=float, default=EPSILON_THRESHOLD)
    parser.add_argument("--patience", type=int,   default=TRACK_PATIENCE,
                        help="Grace period frames before confirming a vehicle left (default 15)")
    args = parser.parse_args()

    processor = VideoProcessor(args.input, args.output, args.model,
                               args.scale, args.epsilon)
    processor.process()


if __name__ == "__main__":
    main()
