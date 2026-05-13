# Vehicle Proximity Analysis – Technical Report

## 1. Logic & Architecture

### Proximity Metric: Why `y2`?
The camera is mounted high above the road and tilted downward. In this perspective, the image coordinate system maps directly to physical depth: objects closer to the camera appear **lower** in the frame (higher pixel `y` value). The bottom edge of a bounding box, `y2`, therefore serves as a reliable, calibration-free proxy for distance from the camera. The vehicle with the highest `y2` at any given moment is the one physically closest to the lens.

### State Management with a Dictionary
The system maintains a `ProximityTracker` object backed by a Python dictionary:

```
track_id  →  { max_y, cpa_location, best_frame, best_timestamp, best_bbox }
```

On every frame, for each detected vehicle the `update()` method compares the current `y2` against the stored `max_y`. If `y2 > max_y`, the record is overwritten. At that moment, the system also computes the **bottom-center contact point** `cpa_location = ((x1+x2)/2, y2)` — the pixel coordinate that best represents where the vehicle's wheels touch the road plane, and the most geometrically stable anchor for any downstream distance estimation. This is an O(1) lookup per vehicle per frame — efficient regardless of the number of tracked objects. Because the tracker holds state between frames (YOLO's `persist=True` ensures stable IDs), each vehicle accumulates a single authoritative CPA record by the end of the video.

---

## 2. System Accuracy

### How Accuracy Is Measured
- **Visual Validation:** The annotated output video shows each bounding box turn green and display a "CPA" badge precisely when the vehicle reaches its recorded maximum `y2`. A reviewer can scrub through the video and visually confirm the highlighted frame matches the vehicle's closest approach.
- **Timestamp Precision:** Timestamps are computed as `frame_index / fps`, giving sub-frame accuracy limited only by the source video's frame rate (typically ±17 ms at 60 fps, ±33 ms at 30 fps).

### Known Failure Modes
| Scenario | Effect |
|---|---|
| **Occlusion** | If a vehicle is temporarily hidden behind another, the tracker may lose its ID. It re-appears with a new ID, splitting the record — the true CPA may be assigned to the wrong entry. |
| **Bounding box jitter** | YOLO detections fluctuate slightly frame-to-frame. A spuriously large box can produce a false CPA. |
| **Varying vehicle heights** | A tall truck and a low sedan at the same physical distance will have different `y2` values. The metric compares a vehicle against *itself* over time (same ID), so cross-vehicle comparisons are unreliable. |
| **Camera tilt / lens distortion** | If the camera is not perfectly nadir, the `y2` proxy degrades. A vehicle moving laterally in the frame may increase `y2` without actually approaching. |

---

## 3. Critical Analysis – What I Would Improve Next

1. **Smooth `y2` with a rolling average.** Bounding box coordinates jitter by a few pixels between frames. Replacing raw `y2` with a 5-frame rolling mean before the CPA comparison would eliminate false positives caused by detector noise without meaningfully delaying CPA detection.

2. **Camera calibration for metric depth.** Given the camera's intrinsic matrix and homography to the road plane, pixel coordinates can be converted to real-world metres. This replaces the heuristic `y2` proxy with a true Euclidean distance and makes results comparable across camera setups.

3. **Re-ID on track loss.** When a vehicle is occluded and reacquires a new ID, a re-identification module (e.g., feature embedding matching) could merge the two track segments, ensuring the CPA record covers the full trajectory.

4. **Multi-lane awareness.** Tagging each detection with its lane (via a pre-defined lane mask) would allow per-lane CPA statistics — useful for traffic flow analysis.

5. **Real-time streaming support.** Replacing the file-based `VideoCapture` source with an RTSP stream and writing results to a message queue (e.g., Redis) would enable live deployment at road-side camera installations.

---

## 4. Additional Task – What I Would Build Next

If given an additional task beyond CPA detection, I would build a **vehicle speed estimator**.

The CPA record already gives us each vehicle's trajectory over time (frame, timestamp, bounding box). With two known ground-truth distances (e.g., lane markings visible in the video calibrated to metres), we can compute a homography that maps pixel coordinates to real-world coordinates. Applying that transform to consecutive `cpa_location` points gives displacement in metres; dividing by elapsed time gives speed in km/h.

This is a natural extension because:
- The tracking infrastructure is already in place (stable IDs, per-frame bbox).
- The `cpa_location` `[cx, y2]` point is already the most geometrically stable anchor for this calculation.
- Speed data per vehicle would make the system genuinely useful for traffic enforcement and flow analysis — a direct operational upgrade over purely proximity-based reporting.
