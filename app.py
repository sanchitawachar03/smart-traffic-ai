import streamlit as st
import cv2
import tempfile
import time
import numpy as np
from detector import TrafficDetector

# Page configuration
st.set_page_config(
    page_title="AI Smart Traffic Density & Live Green Time Calculator",
    page_icon="⏱️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&family=JetBrains+Mono:wght@600;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main-title {
        background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.2rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 2px;
    }
    .sub-title {
        color: #94A3B8;
        text-align: center;
        font-size: 1.05rem;
        margin-bottom: 25px;
    }

    .glass-card {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 22px;
        margin-bottom: 18px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.3);
    }
    .time-badge {
        font-family: 'JetBrains Mono', monospace;
        font-size: 4rem;
        font-weight: 800;
        color: #10B981;
        text-align: center;
        margin: 5px 0;
        line-height: 1.1;
    }
    .stat-badge {
        background: #0F172A;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 12px 14px;
        text-align: center;
    }
    .live-badge {
        display: inline-block;
        background: #EF4444;
        color: white;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: bold;
        letter-spacing: 1px;
        animation: blink 1.5s infinite;
    }
    @keyframes blink {
        0% { opacity: 1; }
        50% { opacity: 0.4; }
        100% { opacity: 1; }
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">⏱️ AI LIVE TRAFFIC DENSITY & DYNAMIC GREEN TIME CALCULATOR</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">YOLOv8 Real-Time Detection • Mobile Live Camera Feed • Proportional Signal Time Allocation</div>', unsafe_allow_html=True)

# ----------------- SIDEBAR -----------------
st.sidebar.markdown("### 🎛️ Input Source Selection")
source_mode = st.sidebar.radio(
    "Choose Video / Camera Input:",
    ["📱 Mobile Live Camera (IP Webcam)", "📁 Upload Video File", "💻 Laptop Webcam"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ AI Model & Controls")

model_choice = st.sidebar.selectbox(
    "AI Vision Model:",
    ["yolov8n.pt (Nano - Super Fast)", "yolov8s.pt (Small - High Accuracy for Mobile Feeds)"],
    index=0
)
selected_model_name = "yolov8n.pt" if "yolov8n" in model_choice else "yolov8s.pt"

conf_thresh = st.sidebar.slider("Detection Confidence", min_value=0.20, max_value=0.85, value=0.35, step=0.05)
frame_skip = st.sidebar.slider("Frame Skip (Processing Speedup)", min_value=1, max_value=4, value=2)

st.sidebar.markdown("---")
simulate_ambulance = st.sidebar.toggle("🚨 Simulate Ambulance Override", value=False)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📐 Space Density Weights:")
st.sidebar.markdown("- 🚗 **Car**: 2.0 pts")
st.sidebar.markdown("- 🏍️ **Bike**: 1.0 pts")
st.sidebar.markdown("- 🚌 **Bus / Truck**: 4.0 pts")
st.sidebar.markdown("- 🚑 **Emergency**: Instant 45s Corridor")

# Load AI Detector
@st.cache_resource
def get_detector(model_file):
    return TrafficDetector(model_file)

detector = get_detector(selected_model_name)
detector.switch_model(selected_model_name)

# ----------------- INPUT HANDLING -----------------
video_source = None
is_live_stream = False

if source_mode == "📱 Mobile Live Camera (IP Webcam)":
    st.markdown("### 📱 Mobile Live Camera Feed Setup")
    col_help1, col_help2 = st.columns([3, 2])
    with col_help1:
        ip_url = st.text_input(
            "Enter Mobile IP Webcam Stream URL:",
            placeholder="e.g., http://192.168.1.15:8080/video or http://192.168.1.15:4747/video"
        )
    with col_help2:
        st.info("💡 **Kaise connect karein:** Mobile me Play Store se free app **'IP Webcam'** ya **'DroidCam'** download karein, 'Start Server' dabayein aur URL yaha paste karein.")
    
    start_live = st.button("🔴 Start Live Mobile Stream", type="primary")
    if start_live and ip_url:
        video_source = ip_url.strip()
        is_live_stream = True

elif source_mode == "💻 Laptop Webcam":
    st.markdown("### 💻 Laptop / USB Webcam Feed")
    cam_index = st.number_input("Camera Index (Default is 0):", min_value=0, max_value=3, value=0)
    start_webcam = st.button("🔴 Start Live Webcam Feed", type="primary")
    if start_webcam:
        video_source = int(cam_index)
        is_live_stream = True

elif source_mode == "📁 Upload Video File":
    st.markdown("### 📹 Upload Traffic Video")
    uploaded_file = st.file_uploader("Upload Lane Video (`.mp4`, `.avi`, `.mov`)", type=["mp4", "avi", "mov", "mkv"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        tfile.flush()
        video_source = tfile.name
        is_live_stream = False

# ----------------- MAIN DISPLAY -----------------
col_stream, col_decision = st.columns([5, 4])

with col_stream:
    if is_live_stream:
        st.markdown('##### 🎥 Live Camera Stream & AI Tracking <span class="live-badge">LIVE FEED</span>', unsafe_allow_html=True)
    else:
        st.markdown('##### 🎥 Video Stream & AI Tracking', unsafe_allow_html=True)
    video_placeholder = st.empty()

with col_decision:
    st.markdown("##### ⏱️ AI Green Light Allocation")
    decision_placeholder = st.empty()
    breakdown_placeholder = st.empty()

if video_source is not None:
    cap = cv2.VideoCapture(video_source)
    
    if not cap.isOpened():
        st.error(f"❌ Camera source open nahi ho paya: `{video_source}`. Please check karein ki Mobile aur Laptop same Wi-Fi pe hain aur URL sahi hai.")
    else:
        stop_stream = st.button("⏹️ Stop Stream / Disconnect")
        step = 0

        while cap.isOpened() and not stop_stream:
            ret, frame = cap.read()
            if not ret:
                if is_live_stream:
                    st.warning("⚠️ Live Stream disconnected or frame dropped.")
                else:
                    st.success("✅ Video Stream Analysis Complete.")
                break

            step += 1
            if step % frame_skip != 0:
                continue

            frame = cv2.resize(frame, (760, 480))

            # AI Detection & ByteTrack
            annotated_frame, counts, total_vehicles, density_score, emergency_detected = detector.process_frame(
                frame, conf_threshold=conf_thresh, emergency_override=simulate_ambulance
            )

            timing_info = detector.calculate_signal_timing(density_score, total_vehicles, emergency_detected)

            # Render Stream
            rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(rgb_frame, channels="RGB", use_column_width=True)

            # Main Green Time Recommendation Card
            color = timing_info['color_hex']
            rec_time = timing_info['recommended_green']
            
            decision_html = f"""
            <div class="glass-card" style="border-left: 6px solid {color};">
                <span style="font-size: 0.85rem; color: #94A3B8; text-transform: uppercase; letter-spacing: 1px; font-weight: 600;">
                    AI RECOMMENDED GREEN SIGNAL TIME
                </span>
                <div class="time-badge" style="color: {color};">
                    {rec_time}<span style="font-size: 2rem;"> SEC</span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 10px; padding-top: 10px; border-top: 1px solid #334155;">
                    <span style="font-size: 0.95rem; color: #ECEFF1;"><b>Traffic State:</b></span>
                    <span style="color: {color}; font-weight: 700; font-size: 1.05rem;">{timing_info['level']}</span>
                </div>
                <p style="font-size: 0.85rem; color: #94A3B8; margin: 8px 0 0 0;">
                    {timing_info['status']}
                </p>
            </div>
            """
            decision_placeholder.markdown(decision_html, unsafe_allow_html=True)

            # Vehicle Breakdown & Comparison Card
            with breakdown_placeholder.container():
                c1, c2, c3 = st.columns(3)
                c1.markdown(f"""<div class="stat-badge">🚗 <b>Cars</b><br><span style="font-size:1.5rem; color:#00E676; font-weight:bold;">{counts['Car']}</span></div>""", unsafe_allow_html=True)
                c2.markdown(f"""<div class="stat-badge">🏍️ <b>Bikes</b><br><span style="font-size:1.5rem; color:#FFB300; font-weight:bold;">{counts['Motorcycle']}</span></div>""", unsafe_allow_html=True)
                c3.markdown(f"""<div class="stat-badge">🚌 <b>Heavy</b><br><span style="font-size:1.5rem; color:#E91E63; font-weight:bold;">{counts['Bus'] + counts['Truck']}</span></div>""", unsafe_allow_html=True)

                diff_from_static = rec_time - 30
                diff_text = f"+{diff_from_static}s extra to clear jam" if diff_from_static > 0 else (f"{diff_from_static}s saved (low traffic)" if diff_from_static < 0 else "Optimal flow")
                diff_color = "#FACC15" if diff_from_static > 0 else "#38BDF8"

                st.markdown(f"""
                <div style="background: #0F172A; border: 1px solid #334155; border-radius: 12px; padding: 14px; margin-top: 14px;">
                    <div style="display: flex; justify-content: space-between; font-size: 0.95rem;">
                        <span>⚡ <b>Total Congestion Score:</b></span>
                        <span style="color:#00E676; font-weight:bold;">{density_score:.1f} pts</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; font-size: 0.95rem; margin-top: 6px;">
                        <span>🕒 <b>Fixed Timer (Traditional):</b></span>
                        <span style="color:#94A3B8;">30 Seconds</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; font-size: 0.95rem; margin-top: 6px;">
                        <span>🎯 <b>AI Dynamic Advantage:</b></span>
                        <span style="color:{diff_color}; font-weight:bold;">{diff_text}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            time.sleep(0.01)

        cap.release()
else:
    with col_stream:
        st.info("👆 Upar se input source select karein (Mobile Live Camera / Video File / Webcam) to start analysis!")
    with col_decision:
        decision_placeholder.markdown("""
        <div class="glass-card" style="text-align: center; color: #94A3B8; padding: 40px 20px;">
            <h3>⏱️ Awaiting Live / Video Input</h3>
            <p>Connect Mobile Camera or upload a video to calculate real-time traffic density & green time.</p>
        </div>
        """, unsafe_allow_html=True)