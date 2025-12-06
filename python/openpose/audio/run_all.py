# run_all.py
# Runs audio.py then openpose_footstrikes.py automatically

import subprocess
import pathlib
import sys

def run_script(script_name):
    print(f"\n==============================")
    print(f" Running {script_name}")
    print(f"==============================\n")

    result = subprocess.run(
        ["python", script_name],
        cwd=pathlib.Path(__file__).parent
    )

    if result.returncode != 0:
        print(f"\n❌ ERROR: {script_name} failed.\n")
        sys.exit(1)

    print(f"\n✅ Finished {script_name}\n")

def main():
    base = pathlib.Path(__file__).parent

    audio_script = base / "audio.py"
    foot_script  = base / "openpose_footstrikes.py"

    if not audio_script.exists():
        raise SystemExit(f"Missing script: {audio_script}")
    if not foot_script.exists():
        raise SystemExit(f"Missing script: {foot_script}")

    # Step 1 — Beat detection
    run_script("audio.py")

    # Step 2 — Footstrike + graph processing
    run_script("openpose_footstrikes.py")

    print("\n🎉 ALL DONE! Beats + Footstrikes + Graph generated.\n")

if __name__ == "__main__":
    main()
