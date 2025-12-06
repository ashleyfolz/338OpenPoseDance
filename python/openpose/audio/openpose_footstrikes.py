import os
import glob
import json
import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import csv

# =======================
VIDEO_PATH = "/Users/ashleyfolz/openpose/python/openpose/audio/video.mov"
JSON_DIR   = "/Users/ashleyfolz/openpose/openpose_output/emily_mov_1_short"
BPM_PATH = "video.beats.csv"

SMOOTH_SIGMA = 2
MIN_STRIKE_SEPARATION = 0.5  # seconds
THRESH_VEL = 0.001
CONF_THRESH = 0.20
# =======================

# ---- Determine FPS from video file (only metadata, no playback)
import cv2
cap0 = cv2.VideoCapture(VIDEO_PATH)
if not cap0.isOpened():
    raise FileNotFoundError(f"Could not open video metadata: {VIDEO_PATH}")
fps = cap0.get(cv2.CAP_PROP_FPS)
frame_w = int(cap0.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap0.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap0.release()

# ---- Collect JSON files (sorted)
json_files = sorted(glob.glob(os.path.join(JSON_DIR, "*_keypoints.json")))
if not json_files:
    raise FileNotFoundError(f"No JSON files found in {JSON_DIR}")

RIGHT_ANKLE = 11
LEFT_ANKLE  = 14
KP_DIM = 3  # x, y, confidence

def pick_best_person(people):
    if not people:
        return None
    best = None
    best_score = -1
    for p in people:
        kps = p.get("pose_keypoints_2d", [])
        confs = kps[2::3]
        if not confs:
            continue
        score = float(np.mean(confs))
        if score > best_score:
            best = p
            best_score = score
    return best

def read_xyc(person, idx):
    kps = person.get("pose_keypoints_2d", [])
    base = idx * KP_DIM
    if len(kps) < base + 3:
        return None
    return (kps[base], kps[base+1], kps[base+2])

left_x, left_y = [], []
right_x, right_y = [], []

for jf in json_files:
    with open(jf, "r") as f:
        data = json.load(f)

    person = pick_best_person(data.get("people", []))

    if person is None:
        if left_x:
            left_x.append(left_x[-1]); left_y.append(left_y[-1])
            right_x.append(right_x[-1]); right_y.append(right_y[-1])
        else:
            left_x.append(0.0); left_y.append(0.0)
            right_x.append(0.0); right_y.append(0.0)
        continue

    la = read_xyc(person, LEFT_ANKLE)
    ra = read_xyc(person, RIGHT_ANKLE)

    if la is None or la[2] < CONF_THRESH:
        lx, ly = (left_x[-1], left_y[-1]) if left_x else (0.0, 0.0)
    else:
        lx, ly = la[0] / frame_w, la[1] / frame_h

    if ra is None or ra[2] < CONF_THRESH:
        rx, ry = (right_x[-1], right_y[-1]) if right_x else (0.0, 0.0)
    else:
        rx, ry = ra[0] / frame_w, ra[1] / frame_h

    left_x.append(lx); left_y.append(ly)
    right_x.append(rx); right_y.append(ry)

# ---- Smooth ankle trajectories
left_y = gaussian_filter1d(np.array(left_y), sigma=SMOOTH_SIGMA)
right_y = gaussian_filter1d(np.array(right_y), sigma=SMOOTH_SIGMA)

# ---- Velocity
left_vel = np.gradient(left_y) * fps
right_vel = np.gradient(right_y) * fps

left_vel_smooth = -gaussian_filter1d(left_vel, sigma=SMOOTH_SIGMA)
right_vel_smooth = -gaussian_filter1d(right_vel, sigma=SMOOTH_SIGMA)

def detect_strike_start(signal_vel, min_frames, threshold=THRESH_VEL):
    frames = []
    last = -min_frames
    for i in range(1, len(signal_vel)):
        if (signal_vel[i-1] < -threshold) and (signal_vel[i] >= -threshold):
            if i - last >= min_frames:
                frames.append(i)
                last = i
    return np.array(frames, int)

min_dist_frames = max(1, int(MIN_STRIKE_SEPARATION * fps))

left_strikes = detect_strike_start(left_vel_smooth, min_dist_frames)
right_strikes = detect_strike_start(right_vel_smooth, min_dist_frames)

# ---- Load beat times
beat_times = []
with open(BPM_PATH, "r") as f:
    reader = csv.reader(f)
    next(reader); next(reader)  # skip headers
    for row in reader:
        beat_times.append(float(row[0].strip()))
beat_times = np.array(beat_times)

# ---- Plot
t = np.arange(len(left_y)) / fps
plt.figure(figsize=(12, 6))
plt.plot(t, left_y, label="Left Ankle Y", color='red', alpha=0.4)
plt.plot(t, right_y, label="Right Ankle Y", color='blue', alpha=0.4)

plt.scatter(left_strikes / fps, left_y[left_strikes], color='red', s=60, label="Left Strike")
plt.scatter(right_strikes / fps, right_y[right_strikes], color='blue', s=60, label="Right Strike")

# Beat vertical lines
for bt in beat_times:
    plt.axvline(bt, color='black', linewidth=1.2, alpha=0.9)

plt.xlabel("Time (s)")
plt.ylabel("Normalized Vertical Position")
plt.title("Foot Strikes + Beat Timing")
plt.legend()
plt.grid(True)
plt.tight_layout()
graph_path = "footstrike_graph.png"
plt.savefig(graph_path)
plt.close()
print(f"\n✅ Saved graph to {graph_path}\n")

# ---- Optional Output CSV
with open("foot_strikes.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["side", "frame", "time_s"])
    for fr in left_strikes:
        w.writerow(["left", fr, fr / fps])
    for fr in right_strikes:
        w.writerow(["right", fr, fr / fps])

print("\n✅ Finished graph + foot strike CSV.\n")
