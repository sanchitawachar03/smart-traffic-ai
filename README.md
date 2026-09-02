# 🚦 AI-Powered Adaptive Traffic Signal Control System

Computer Vision (YOLOv8) based Real-Time Traffic Density Detection & Dynamic Signal Allocation.

---

## 📂 Project Files
- pp.py : Streamlit Web UI Dashboard (Upload Video, Live Visual Bounding Boxes, Dynamic Timers, Emergency Override).
- detector.py : Core YOLOv8 AI Detection & Density Logic.
- un_cli.py : Direct OpenCV Desktop Window mode (python run_cli.py).
- equirements.txt : Python libraries list.

---

## 🚀 How to Run

### Step 1: Open Terminal / CMD in this folder
1. Open Command Prompt (cmd) or PowerShell.
2. Navigate to this folder:
   `cmd
   cd "C:\Traffic Signal AIML"
   `

### Step 2: Install Required Libraries (One-time)
`cmd
pip install -r requirements.txt
`

### Step 3: Run the Web Dashboard (Recommended for College Project Demo)
`cmd
streamlit run app.py
`
👉 Browser will automatically open at http://localhost:8501.

---

### Alternative: Run directly with OpenCV Desktop Window
`cmd
python run_cli.py
`
*(Or specify a video file: python run_cli.py my_video.mp4)*
- Press **'q'** to quit.
- Press **'e'** to toggle Emergency Ambulance simulation.
