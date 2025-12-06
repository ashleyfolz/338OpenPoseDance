import streamlit as st
import subprocess
import pathlib
import os

# === Directories (same as before) ===
AUDIO_DIR = pathlib.Path(__file__).parent
UPLOADS_DIR = AUDIO_DIR / "uploads"
UPLOADS_DIR.mkdir(exist_ok=True)

OPENPOSE_ROOT = pathlib.Path("/Users/ashleyfolz/openpose")
JSON_BASE = OPENPOSE_ROOT / "openpose_output"

AUDIO_SCRIPT = AUDIO_DIR / "audio.py"
FOOTSTRIKE_SCRIPT = AUDIO_DIR / "openpose_footstrikes.py"

st.title("Deep Dance Dashboard 💃🕺")

# -----------------------------
# SIDEBAR: Detect existing OpenPose folders
# -----------------------------
st.sidebar.title("OpenPose Results")

existing_folders = [
    f.name for f in JSON_BASE.iterdir()
    if f.is_dir() and any(f.glob("*_keypoints.json"))
]

if existing_folders:
    selected_folder = st.sidebar.selectbox(
        "Choose an existing OpenPose run:",
        existing_folders
    )
else:
    selected_folder = None
    st.sidebar.write("No OpenPose output folders yet.")


# -----------------------------
# Main UI: Upload new video
# -----------------------------
st.header("Upload New Video")
uploaded = st.file_uploader("Upload a dance video", type=["mp4", "mov"])

if uploaded:
    video_path = UPLOADS_DIR / uploaded.name
    with open(video_path, "wb") as f:
        f.write(uploaded.read())

    st.success(f"Uploaded video: {uploaded.name}")
    st.session_state["video_path"] = video_path
    st.session_state["video_stem"] = video_path.stem


# -----------------------------
# STEP 1: Run OpenPose
# -----------------------------
st.header("Run OpenPose")

if "video_path" in st.session_state:
    if st.button("Run OpenPose on Uploaded Video"):
        vid = st.session_state["video_path"]

        json_output_dir = JSON_BASE / vid.stem
        json_output_dir.mkdir(exist_ok=True)

        # path for rendered OpenPose video (MP4 with audio)
        rendered_video = json_output_dir / f"{vid.stem}_openpose.mp4"

        cmd = [
            "./build/examples/openpose/openpose.bin",
            "--video", str(vid.resolve()),
            "--write_json", str(json_output_dir.resolve()),
            "--write_video", str(rendered_video.resolve()),
            "--write_video_with_audio",
            "--render_pose", "1",
            "--display", "0"
        ]

        subprocess.run(cmd, cwd=str(OPENPOSE_ROOT))

        st.success(f"OpenPose completed → {json_output_dir}")

        # 🎥 Show & download the rendered video if it exists
        if rendered_video.exists():
            st.subheader("Rendered OpenPose Video (This Run)")
            st.video(str(rendered_video))
            with open(rendered_video, "rb") as vf:
                st.download_button(
                    label="Download OpenPose Video",
                    data=vf,
                    file_name=rendered_video.name,
                    mime="video/mp4"
                )
        else:
            st.warning("Rendered OpenPose video was not created.")
else:
    st.info("Upload a video to enable OpenPose.")


# -----------------------------
# STEP 2: Beat Detection (Sidebar option OR new upload)
# -----------------------------
st.header("Beat Detection")

if selected_folder:
    st.write(f"Selected OpenPose folder: **{selected_folder}**")

    # Look for matching video in uploads/
    possible_video = UPLOADS_DIR / f"{selected_folder}.mov"
    if not possible_video.exists():
        possible_video = UPLOADS_DIR / f"{selected_folder}.mp4"

    if possible_video.exists():
        st.write(f"Matched video: {possible_video.name}")
    else:
        st.error("No matching video found in uploads/.")
        possible_video = None

    beat_csv = AUDIO_DIR / f"{selected_folder}.beats.csv"

    if possible_video and st.button("Run Beat Detection on Selected Folder"):
        cmd = [
            "python",
            str(AUDIO_SCRIPT),
            "--video", str(possible_video.resolve()),
            "--out", str(beat_csv.resolve())
        ]

        subprocess.run(cmd)
        st.success(f"Beat detection complete → {beat_csv}")

    # 🔽 Download button for beats CSV (if it exists)
    if beat_csv.exists():
        with open(beat_csv, "rb") as f:
            st.download_button(
                label="Download Beat CSV",
                data=f,
                file_name=f"{selected_folder}.beats.csv",
                mime="text/csv"
            )

else:
    st.info("Select an OpenPose folder from the sidebar to run beat detection.")


# -----------------------------
# STEP 3: Footstrike Graph
# -----------------------------
st.header("Footstrike Graph")

if selected_folder:
    json_dir = JSON_BASE / selected_folder
    beat_csv = AUDIO_DIR / f"{selected_folder}.beats.csv"
    strike_graph = AUDIO_DIR / "footstrike_graph.png"
    strikes_csv = AUDIO_DIR / "foot_strikes.csv"  # from openpose_footstrikes.py

    if st.button("Generate Footstrike Graph for Selected Folder"):
        # must have matching video
        possible_video = UPLOADS_DIR / f"{selected_folder}.mp4"
        if not possible_video.exists():
            possible_video = UPLOADS_DIR / f"{selected_folder}.mov"

        cmd = [
            "python",
            str(FOOTSTRIKE_SCRIPT),
            "--json_dir", str(json_dir.resolve()),
            "--video", str(possible_video.resolve()),
            "--beats", str(beat_csv.resolve())
        ]

        subprocess.run(cmd)

        if strike_graph.exists():
            st.image(str(strike_graph))
            st.success("Footstrike graph generated!")
        else:
            st.error("No graph image found. Did the script save it?")

    # 🔽 Download button for foot_strikes.csv (if it exists)
    if strikes_csv.exists():
        with open(strikes_csv, "rb") as f:
            st.download_button(
                label="Download Footstrike CSV",
                data=f,
                file_name=f"{selected_folder}_foot_strikes.csv",
                mime="text/csv"
            )
else:
    st.info("Select an OpenPose folder from the sidebar to generate footstrike graph.")


# -----------------------------
# STEP 4: View Rendered Video for Existing Runs
# -----------------------------
st.header("Rendered OpenPose Video (Existing Run)")

if selected_folder:
    existing_rendered = JSON_BASE / selected_folder / f"{selected_folder}_openpose.mp4"
    if existing_rendered.exists():
        st.video(str(existing_rendered))
        with open(existing_rendered, "rb") as vf:
            st.download_button(
                label="Download OpenPose Video for Selected Run",
                data=vf,
                file_name=existing_rendered.name,
                mime="video/mp4"
            )
    else:
        st.info("No rendered OpenPose video found for this run yet. Run OpenPose again on the matching upload to create one.")
else:
    st.info("Select an OpenPose run in the sidebar to preview its rendered video.")
