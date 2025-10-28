import argparse, subprocess, os, sys
from pathlib import Path

# repo root = .../openpose (this file is .../openpose/python/openpose/audio/rundual.py)
ROOT = Path(__file__).resolve().parents[3]   # go up from audio -> openpose/python -> openpose
AUDIO_DIR = Path(__file__).resolve().parent  # .../python/openpose/audio
print("[debug] ROOT =", ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Path to input video (relative to repo root or absolute)")
    ap.add_argument("--openpose_bin", default=str(ROOT / "build/examples/openpose/openpose.bin"))
    ap.add_argument("--op_out_dir", default=str(ROOT / "openpose_output"))
    ap.add_argument("--audio_env", default="deepdance")
    ap.add_argument("--audio_script", default=str(AUDIO_DIR / "audio.py"))
    args = ap.parse_args()

    # absolutize paths
    video = Path(args.video)
    if not video.is_absolute():
        video = (ROOT / video).resolve()
    op_bin = Path(args.openpose_bin).resolve()
    op_out = (Path(args.op_out_dir) / video.stem).resolve()
    audio_script = Path(args.audio_script).resolve()

    # sanity checks
    if not op_bin.exists():
        sys.exit(f"[error] OpenPose binary not found: {op_bin}")
    if not video.exists():
        sys.exit(f"[error] video not found: {video}")
    if not audio_script.exists():
        sys.exit(f"[error] audio script not found: {audio_script}")

    # dirs
    (ROOT / "logs").mkdir(exist_ok=True)
    op_out.mkdir(parents=True, exist_ok=True)
    beats_csv = (ROOT / "data/beats" / f"{video.stem}.beats.csv")
    beats_csv.parent.mkdir(parents=True, exist_ok=True)

    # commands
    openpose_cmd = [
        str(op_bin),
        "--video", str(video),
        "--write_json", str(op_out),
        "--display", "0",
        "--render_pose", "0",
    ]
    audio_cmd = [
        "conda", "run", "-n", args.audio_env, "python",
        str(audio_script),
        "--video", str(video),
        "--out", str(beats_csv),
    ]

    print("Launching OpenPose:", " ".join(openpose_cmd))
    p1 = subprocess.Popen(openpose_cmd,
                          stdout=open(ROOT / "logs" / f"{video.stem}_openpose.log", "w"),
                          stderr=subprocess.STDOUT)

    print("Launching audio:", " ".join(audio_cmd))
    p2 = subprocess.Popen(audio_cmd,
                          stdout=open(ROOT / "logs" / f"{video.stem}_audio.log", "w"),
                          stderr=subprocess.STDOUT)

    rc1 = p1.wait()
    rc2 = p2.wait()
    if rc1 != 0 or rc2 != 0:
        sys.exit(f"Error: openpose rc={rc1}, audio rc={rc2}")
    print("✅ done")

if __name__ == "__main__":
    main()
