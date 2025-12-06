# audio/audio.py
# Beat detection (works with or without CLI args)

import os, csv, pathlib, contextlib, tempfile, argparse
from moviepy.editor import VideoFileClip
import librosa

# ---------- audio utils ----------
def extract_audio_wav(video_path, target_sr=22050):
    """Returns mono audio y and sample rate sr from video. Uses a temp WAV."""
    tmp_wav = None
    with VideoFileClip(str(video_path)) as clip:
        folder = pathlib.Path(video_path).parent
        fd, tmp_wav = tempfile.mkstemp(
            prefix="video_audio_", suffix=".wav", dir=folder
        )
        os.close(fd)
        clip.audio.write_audiofile(
            tmp_wav, fps=target_sr, nbytes=2,
            codec="pcm_s16le", verbose=False, logger=None
        )

    y, sr = librosa.load(tmp_wav, sr=target_sr, mono=True)

    with contextlib.suppress(Exception):
        os.remove(tmp_wav)

    return y, sr

def detect_beats(y, sr, hop_length=512):
    oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    tempo, beat_frames = librosa.beat.beat_track(
        onset_envelope=oenv, sr=sr, hop_length=hop_length, units="frames"
    )
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)
    return float(tempo), beat_times

# ---------- CSV ----------
def write_beats_csv(out_path, tempo_bpm, beat_times):
    out_p = pathlib.Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tempo_bpm", f"{tempo_bpm:.4f}"])
        w.writerow(["beat_time_s"])
        for t in beat_times:
            w.writerow([f"{t:.6f}"])
    return out_p

# ---------- MAIN ----------
def main():
    print("\n=== Running Beat Detection (audio.py) ===")

    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default=None, help="Path to input video")
    parser.add_argument("--out", default=None, help="Path to output CSV")
    args = parser.parse_args()

    script_dir = pathlib.Path(__file__).parent

    # default video if none passed
    if args.video is None:
        video_path = script_dir / "video.mov"
    else:
        video_path = pathlib.Path(args.video)

    if not video_path.exists():
        raise SystemExit(f"[ERROR] video not found: {video_path}")

    print(f"Using video: {video_path}")

    # default out if none passed
    if args.out is None:
        out_csv = script_dir / f"{video_path.stem}.beats.csv"
    else:
        out_csv = pathlib.Path(args.out)

    print(f"Writing beats to: {out_csv}")

    # Extract audio
    y, sr = extract_audio_wav(video_path)

    # Detect beats
    tempo_bpm, beat_times = detect_beats(y, sr)

    # Write CSV
    out_file = write_beats_csv(out_csv, tempo_bpm, beat_times)

    print(f"Estimated tempo: {tempo_bpm:.1f} BPM")
    print(f"Detected {len(beat_times)} beats")
    print(f"Wrote beats to → {out_file}")
    print("=========================================\n")

if __name__ == "__main__":
    main()
