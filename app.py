"""
Streamlit Web Dashboard for Real-Time Head & Person Counter.
Provides an interactive UI for camera feeds, uploaded video analysis, and real-time analytics.
"""

import os
import tempfile
import time
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from counter import HeadCounter


def draw_person_silhouette(img, x, y, scale=1.0):
    """Draw a basic person figure with head, torso, and legs."""
    head_r = int(18 * scale)
    head_cy = y - int(65 * scale)
    head_cx = x

    # Head
    cv2.circle(img, (head_cx, head_cy), head_r, (210, 210, 210), -1)
    # Torso
    cv2.rectangle(
        img,
        (x - int(22 * scale), head_cy + head_r),
        (x + int(22 * scale), y + int(25 * scale)),
        (180, 180, 180),
        -1,
    )
    # Legs
    cv2.line(
        img,
        (x - int(12 * scale), y + int(25 * scale)),
        (x - int(15 * scale), y + int(70 * scale)),
        (160, 160, 160),
        int(8 * scale),
    )
    cv2.line(
        img,
        (x + int(12 * scale), y + int(25 * scale)),
        (x + int(15 * scale), y + int(70 * scale)),
        (160, 160, 160),
        int(8 * scale),
    )


def generate_synthetic_demo_video(filename="demo_sample.mp4", num_frames=100):
    """Generate a synthetic test video with moving figures to test counting logic."""
    w, h = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(filename, fourcc, 24.0, (w, h))

    for i in range(num_frames):
        frame = np.full((h, w, 3), 35, dtype=np.uint8)
        # Background floor lines
        for gy in range(220, h, 40):
            cv2.line(frame, (0, gy), (w, gy), (55, 55, 55), 1)

        # Person 1 walking from left to right
        p1_x = int((i * 5.0) % (w + 100)) - 50
        p1_y = 260
        draw_person_silhouette(frame, p1_x, p1_y, scale=1.0)

        # Person 2 walking from right to left
        p2_x = int(w - ((i * 4.0) % (w + 100))) + 50
        p2_y = 250
        draw_person_silhouette(frame, p2_x, p2_y, scale=1.1)

        # Person 3 standing
        if 15 <= i <= 85:
            draw_person_silhouette(frame, 320, 240, scale=0.95)

        writer.write(frame)

    writer.release()
    return filename


@st.cache_resource
def load_counter(model_name: str):
    """Cache the YOLO model instance."""
    return HeadCounter(model_name=model_name)


def main():
    st.set_page_config(
        page_title="Head & Person Counter AI",
        page_icon="👥",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        .metric-card {
            background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
            border: 1px solid #374151;
            border-radius: 12px;
            padding: 16px;
            text-align: center;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
        }
        .metric-val {
            font-size: 2.2rem;
            font-weight: 700;
            color: #10b981;
            margin: 4px 0;
        }
        .metric-title {
            font-size: 0.9rem;
            color: #9ca3af;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("🎯 Real-Time Head & Person Counter")
    st.caption("AI-powered computer vision prototype for counting heads and persons in live camera streams and video recordings.")

    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Configuration")

        source_type = st.radio(
            "Select Video Source",
            ["Upload Video File", "Live Webcam / Camera", "Demo Sample Video"],
            index=0,
        )

        st.subheader("Model & Detection Settings")
        model_choice = st.selectbox(
            "YOLO Model",
            ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"],
            index=0,
            help="yolov8n is ultra-fast for CPU; yolov8s/m provide higher precision.",
        )

        detect_mode = st.radio(
            "Detection Target",
            ["head", "person"],
            format_func=lambda x: "👤 Head Region (Upper Body/Head)" if x == "head" else "🚶 Full Person Body",
            index=0,
        )

        conf_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.10,
            max_value=0.90,
            value=0.35,
            step=0.05,
            help="Higher values reduce false positives; lower values detect distant/partially obscured heads.",
        )

        webcam_idx = 0
        if source_type == "Live Webcam / Camera":
            webcam_idx = st.number_input("Camera Device Index", min_value=0, max_value=5, value=0, step=1)

    # Initialize counter
    counter = load_counter(model_choice)

    # Metric placeholders
    col1, col2, col3, col4 = st.columns(4)
    metric_count = col1.empty()
    metric_peak = col2.empty()
    metric_fps = col3.empty()
    metric_frames = col4.empty()

    def update_metrics(current_count, peak_count, fps, frame_no):
        metric_count.markdown(
            f'<div class="metric-card"><div class="metric-title">Current In-Frame</div><div class="metric-val">{current_count}</div></div>',
            unsafe_allow_html=True,
        )
        metric_peak.markdown(
            f'<div class="metric-card"><div class="metric-title">Peak Detected</div><div class="metric-val" style="color:#60a5fa">{peak_count}</div></div>',
            unsafe_allow_html=True,
        )
        metric_fps.markdown(
            f'<div class="metric-card"><div class="metric-title">Speed (FPS)</div><div class="metric-val" style="color:#f59e0b">{fps:.1f}</div></div>',
            unsafe_allow_html=True,
        )
        metric_frames.markdown(
            f'<div class="metric-card"><div class="metric-title">Processed Frames</div><div class="metric-val" style="color:#a78bfa">{frame_no}</div></div>',
            unsafe_allow_html=True,
        )

    # Initial metric display
    update_metrics(0, 0, 0.0, 0)
    st.divider()

    main_col, chart_col = st.columns([1.6, 1])
    video_placeholder = main_col.empty()
    chart_placeholder = chart_col.empty()

    # Mode 1: Upload Video File
    if source_type == "Upload Video File":
        uploaded_file = st.file_uploader(
            "Upload a video file (.mp4, .avi, .mov, .mkv)",
            type=["mp4", "avi", "mov", "mkv"],
        )

        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_file.read())
            video_path = tfile.name
            tfile.close()

            start_btn = st.button("🚀 Start Video Analysis", type="primary")

            if start_btn:
                run_video_processing(
                    video_path,
                    counter,
                    conf_threshold,
                    detect_mode,
                    video_placeholder,
                    chart_placeholder,
                    update_metrics,
                )

    # Mode 2: Live Webcam
    elif source_type == "Live Webcam / Camera":
        run_cam = st.toggle("🔴 Turn ON Live Camera", value=False)

        if run_cam:
            cap = cv2.VideoCapture(int(webcam_idx))
            if not cap.isOpened():
                st.error(f"Failed to access webcam at index {webcam_idx}. Check your camera connection or permissions.")
            else:
                peak_count = 0
                frame_count = 0
                count_history = []
                stop_btn = st.button("⏹️ Stop Stream")

                while run_cam and not stop_btn:
                    ret, frame = cap.read()
                    if not ret:
                        st.warning("Failed to grab frame from camera.")
                        break

                    frame_count += 1
                    annotated_frame, count, _ = counter.process_frame(
                        frame,
                        conf_threshold=conf_threshold,
                        detect_mode=detect_mode,
                        draw=True,
                    )

                    peak_count = max(peak_count, count)
                    count_history.append(count)
                    if len(count_history) > 60:
                        count_history.pop(0)

                    # Update UI
                    rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    video_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

                    update_metrics(count, peak_count, counter.fps, frame_count)

                    if len(count_history) > 1:
                        df_history = pd.DataFrame({"Frame": range(len(count_history)), "Head Count": count_history})
                        chart_placeholder.line_chart(df_history.set_index("Frame"), height=260)

                cap.release()

    # Mode 3: Demo Sample Video
    elif source_type == "Demo Sample Video":
        st.info("Generates a synthetic animated motion test video with moving person silhouettes to test counting logic.")
        if st.button("🎬 Generate & Run Demo"):
            demo_path = generate_synthetic_demo_video()
            run_video_processing(
                demo_path,
                counter,
                conf_threshold,
                detect_mode,
                video_placeholder,
                chart_placeholder,
                update_metrics,
            )


def run_video_processing(
    video_path,
    counter,
    conf_threshold,
    detect_mode,
    video_placeholder,
    chart_placeholder,
    update_metrics,
):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0

    out_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    out_path = out_temp.name
    out_temp.close()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps_in, (width, height))

    progress_bar = st.progress(0, text="Processing video frames...")
    peak_count = 0
    frame_idx = 0
    count_history = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        annotated_frame, count, _ = counter.process_frame(
            frame,
            conf_threshold=conf_threshold,
            detect_mode=detect_mode,
            draw=True,
        )

        peak_count = max(peak_count, count)
        count_history.append(count)
        writer.write(annotated_frame)

        # Update preview every few frames for responsiveness
        if frame_idx % 2 == 0 or frame_idx == total_frames:
            rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)
            update_metrics(count, peak_count, counter.fps, frame_idx)

            progress_val = min(1.0, frame_idx / total_frames)
            progress_bar.progress(progress_val, text=f"Processing frame {frame_idx}/{total_frames}...")

            if len(count_history) > 2:
                df_history = pd.DataFrame({"Frame": range(len(count_history)), "Head Count": count_history})
                chart_placeholder.line_chart(df_history.set_index("Frame"), height=260)

    cap.release()
    writer.release()
    progress_bar.empty()
    st.success("✅ Video processing complete!")

    if os.path.exists(out_path):
        with open(out_path, "rb") as f:
            st.download_button(
                label="📥 Download Annotated Video",
                data=f.read(),
                file_name="annotated_head_count.mp4",
                mime="video/mp4",
            )


if __name__ == "__main__":
    main()
