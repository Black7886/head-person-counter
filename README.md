# 🎯 Real-Time Head & Person Counter Prototype

An AI-driven Computer Vision application designed to accurately detect and count heads/persons in real-time from **live webcam feeds** and **pre-recorded video files**.

---

## ✨ Features

- **Real-Time Detection & Counting**: Detects heads/people in real-time with instant HUD counts and bounding boxes.
- **Dual Source Support**:
  - **Live Camera**: Direct webcam / external USB camera / IP RTSP stream.
  - **Video Files**: Supports `.mp4`, `.avi`, `.mov`, `.mkv`.
- **Flexible Detection Targets**:
  - `Head Region`: Isolates and bounds head/upper-body regions.
  - `Full Person`: Bounds entire individual.
- **Analytics & HUD**:
  - Live in-frame count, peak count, running frame rate (FPS), and time-series line chart.
- **Dual Interfaces**:
  - **Interactive Streamlit Web Dashboard** (`app.py`).
  - **Fast Lightweight Command-Line Interface** (`detect.py`).

---

## 🚀 Getting Started

### 1. Activate Environment

Open PowerShell or Command Prompt in the project folder:

```powershell
cd "C:\Users\Ayan Gawali\.gemini\antigravity\scratch\head_counter_prototype"
..\venv\Scripts\activate
```

---

### 2. Option A: Run the Web Dashboard (Recommended)

Launch the interactive web UI:

```powershell
streamlit run app.py
```

- Choose **"Upload Video File"** or **"Live Webcam / Camera"** in the sidebar.
- Adjust the **Confidence Threshold** slider (default: `0.35`).
- View real-time charts and download the annotated video.

---

### 3. Option B: Run via Command-Line (OpenCV Window)

#### Using your Webcam (Live Camera):
```powershell
python detect.py --source 0
```

#### Using a Video File:
```powershell
python detect.py --source "path/to/your/video.mp4"
```

#### Save the Annotated Video Output:
```powershell
python detect.py --source "path/to/your/video.mp4" --save "output_annotated.mp4"
```

#### Custom Confidence & Full Body Mode:
```powershell
python detect.py --source 0 --conf 0.40 --mode person
```

*Note: Press **`q`** or **`ESC`** on the video window to quit.*

---

## ⚙️ Command-Line Options

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--source` | `str/int` | `0` | Camera device index (`0`) or video file path |
| `--conf` | `float` | `0.35` | Confidence threshold (0.0 to 1.0) |
| `--mode` | `str` | `head` | `head` or `person` |
| `--model` | `str` | `yolov8n.pt` | YOLO model name (`yolov8n.pt`, `yolov8s.pt`, etc.) |
| `--save` | `str` | `None` | Path to save processed output video |
| `--no-show`| `flag` | `False` | Run headlessly without popping up a GUI window |

---

## 📁 Project Structure

```
head_counter_prototype/
├── counter.py          # Core YOLO detection & HUD drawing engine
├── detect.py           # CLI tool with OpenCV live display
├── app.py              # Streamlit Web UI dashboard
├── requirements.txt    # Python dependencies
└── README.md           # Documentation and quickstart guide
```
