# audio/audio.py
# Beat detection & (optional) basic alignment utilities

import os, csv, argparse, pathlib, contextlib, tempfile
import numpy as np
from moviepy.editor import VideoFileClip
import librosa
from scipy.spatial import cKDTree

# ---------- audio utils ----------
def extract_audio_wav(video_path, target_sr=22050):
    """Returns mono audio y and sample rate sr from video. Uses a temp WAV."""
    tmp_wav = None
    with VideoFileClip(video_path) as clip:
        # write temp wav beside the video to avoid permissions issues
        d = pathlib.Path(video_path).parent
        fd, tmp_wav = tempfile.mkstemp(prefix=pathlib.Path(video_path).stem + "_", suffix=".wav", dir=d)
        os.close(fd)
        clip.audio.write_audiofile(tmp_wav, fps=target_sr, nbytes=2, codec="pcm_s16le",
                                   verbose=False, logger=None)
    y, sr = librosa.load(tmp_wav, sr=target_sr, mono=True)
    # cleanup
    with contextlib.suppress(Exception):
        os.remove(tmp_wav)
    return y, sr

def detect_beats(y, sr, hop_length=512):
    oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    tempo, beat_frames = librosa.beat.beat_track(onset_envelope=oenv, sr=sr, hop_length=hop_length, units='frames')
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)
    return float(tempo), beat_times, oenv

# ---------- timing helpers ----------
def frame_times_from_fps(n_frames, fps):
    return np.arange(n_frames, dtype=float) / float(fps)

def nearest_beat_errors(event_times, beat_times):
    if len(beat_times) == 0 or len(event_times) == 0:
        return np.array([]), np.array([])
    tree = cKDTree(beat_times[:, None])
    dists, idxs = tree.query(event_times[:, None], k=1)
    nearest = beat_times[idxs]
    errors_ms = (event_times - nearest) * 1000.0  # negative = ahead of beat
    return nearest, errors_ms

def estimate_best_lag(event_times, beat_times, search_ms=300):
    if len(event_times) == 0 or len(beat_times) == 0:
        return 0.0
    lags = np.linspace(-search_ms, search_ms, 121)
    best_lag, best_score = 0.0, np.inf
    for lag in lags:
        shifted = event_times + (lag / 1000.0)
        _, errs = nearest_beat_errors(shifted, beat_times)
        score = np.median(np.abs(errs)) if len(errs) else np.inf
        if score < best_score:
            best_score, best_lag = score, float(lag)
    return best_lag

# ---------- csv io ----------
def write_beats_csv(out_path, tempo_bpm, beat_times):
    out_p = pathlib.Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tempo_bpm", f"{tempo_bpm:.4f}"])
        w.writerow(["beat_time_s"])
        for t in beat_times:
            w.writerow([f"{t:.6f}"])
    return str(out_p)

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Extract audio from video, detect beats, and write beat times.")
    ap.add_argument("--video", required=True, help="Path to input video (e.g., videos/example.mp4)")
    ap.add_argument("--out", required=True, help="Output CSV path for beats (e.g., data/beats/example.beats.csv)")
    ap.add_argument("--sr", type=int, default=22050, help="Target sample rate for analysis (default 22050)")
    ap.add_argument("--print", dest="do_print", action="store_true", help="Print summary to stdout")
    ap.add_argument("--demo-align", action="store_true",
                    help="Run demo alignment using synthetic event times (for quick sanity check)")
    args = ap.parse_args()

    video_path = args.video
    if not os.path.isfile(video_path):
        raise SystemExit(f"[error] video not found: {video_path}")

    # 1) audio
    y, sr = extract_audio_wav(video_path, target_sr=args.sr)

    # 2) beats
    tempo_bpm, beat_times, _ = detect_beats(y, sr)
    out_csv = write_beats_csv(args.out, tempo_bpm, beat_times)

    if args.do_print:
        print(f"Estimated tempo: {tempo_bpm:.1f} BPM | Beats: {len(beat_times)}")
        print(f"Wrote beats → {out_csv}")

    # 3) optional demo alignment (kept from your example)
    if args.demo_align:
        # Use actual video fps/frames for realism (no extra deps; use moviepy metadata)
        with VideoFileClip(video_path) as clip:
            fps = clip.fps or 30.0
            n_frames = int(round((clip.duration or 0) * fps))
        frame_ts = frame_times_from_fps(max(n_frames, 1), fps)

        # dummy foot-strikes like your example
        event_times = np.arange(1.0, min(30.0, (len(frame_ts) / fps) - 0.5), 0.5)
        lag_ms = estimate_best_lag(event_times, beat_times, search_ms=250)
        nearest, errors_ms = nearest_beat_errors(event_times + lag_ms / 1000.0, beat_times)
        if len(errors_ms):
            print(f"[demo] Applied lag: {lag_ms:.1f} ms")
            print(f"[demo] Median |error|: {np.median(np.abs(errors_ms)):.1f} ms")
            print(f"[demo] Mean signed error (neg=ahead): {np.mean(errors_ms):.1f} ms")

if __name__ == "__main__":
    main()
