import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from ultralytics import YOLO
from PIL import Image
import tempfile
import cv2
import numpy as np
import base64
import time
import tempfile
import os

# Load YOLO model
model = YOLO("model/traffic_sign_model.pt")

confidence = st.sidebar.slider("Confidence", min_value=0.0, max_value=1.0, value=0.5, step=0.1)

st.title("Object Detection Demo")

# --- Session state ---
if "detected_class" not in st.session_state:
    st.session_state.detected_class = None
if "last_played" not in st.session_state:
    st.session_state.last_played = None

def play_audio(class_name: str):
    match class_name:
        case 'T-intersection':
            file_name = "T intersection.mp3"
        case "cross road":
            file_name = "cross road.mp3"
        case "do not enter":
            file_name = "do not enter.mp3"
        case "keep right":
            file_name = "keep right.mp3"
        case "left curve ahead":
            file_name = "left curve ahead.mp3"
        case "motorcycle crossing":
            file_name = "motor cycle crossing.mp3"
        case "no parking":
            file_name = "no parking.mp3"
        case "no trucks":
            file_name = "no trucks.mp3"
        case "no u-turn":
            file_name = "no u turn.mp3"
        case "right curve ahead":
            file_name = "right curve ahead.mp3"
        case "roundabout":
            file_name = "roundabout.mp3"
        case "school zone":
            file_name = "school zone.mp3"
        case "speed limit":
            file_name = "speed limit.mp3"
        case "steep hill ahead":
            file_name = "steep hill ahead.mp3"
        case "stop":
            file_name = "stop.mp3"
        case "traffic light ahead":
            file_name = "traffic light ahead.mp3"
        case "winding road ahead":
            file_name = "winding road ahead.mp3"
    autoplay_audio("audio/" + file_name)

audio_slot = st.empty()

def autoplay_audio(file_path: str):
    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        md = f"""
                <audio id="auto_audio" autoplay>
                    <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
                </audio>
                <script>
                    var audio = document.getElementById("auto_audio");
                    audio.play();
                </script>
            """
        audio_slot.markdown(md, unsafe_allow_html=True)

# Session state for webcam
if "webcam_running" not in st.session_state:
    st.session_state.webcam_running = False

# Input type selection
img_tab, video_tab, cam_tab, web_rtc, savedVideo = st.tabs(["Image", "Video", "Webcam", "WebRtc", "Saved Video"])

# -------------------- IMAGE --------------------
with img_tab:
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        img = Image.open(uploaded_file)
        results = model(np.array(img), conf=confidence)[0]  # YOLO accepts numpy array
        result_image = results.plot()
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])   # class ID
                class_name = model.names[cls_id]  # class name
                conf = float(box.conf[0]) # confidence score

            print(f"Detected {class_name} with confidence {conf:.2f}")
            play_audio(class_name)

        st.image(result_image, caption="Uploaded Image", use_container_width=True)

# -------------------- VIDEO --------------------
with video_tab:
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])
    if uploaded_file:
        # Save uploaded video temporarily
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

        cap = cv2.VideoCapture(temp_file_path)

        st_frame = st.empty()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame, conf=confidence)

            annotated_frame = results[0].plot()

            st_frame.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), channels = "RGB")

        cap.release()

# -------------------- WEBCAM --------------------
with cam_tab:
    st.write("Webcam streaming using OpenCV")

    col1, col2 = st.columns([1,2])
    with col1:
        start_button = st.button("Start Webcam", type="primary", key="start_webcam")
    with col2:
        stop_button = st.button("Stop Webcam", type="secondary", key="stop_webcam")

    if start_button:
        st.session_state.webcam_running = True
    if stop_button:
        st.session_state.webcam_running = False

    if st.session_state.webcam_running:
        cap = cv2.VideoCapture(0)
        frame_slot = st.empty()  # placeholder for updating frames

        while st.session_state.webcam_running:
            ret, frame = cap.read()
            if not ret:
                st.warning("Failed to capture webcam frame.")
                break

            # YOLO detection
            results = model(frame, conf=confidence)[0]

            # Draw bounding boxes with labels
            if hasattr(results, "boxes") and len(results.boxes) > 0:
                for i, box in enumerate(results.boxes.xyxy):
                    x1, y1, x2, y2 = map(int, box)
                    cls_id = int(results.boxes.cls[i])          # class index
                    conf = float(results.boxes.conf[i])         # confidence
                    label = f"{model.names[cls_id]} {conf:.2f}"

                    # Draw rectangle
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # Draw label background
                    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                    cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), (0, 255, 0), -1)
                    # Put label text
                    cv2.putText(frame, label, (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

            # Show frame in Streamlit
            frame_slot.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        cap.release()

with web_rtc:
    # Transformer class for real-time detection
    class YOLOTransformer(VideoTransformerBase):
        def __init__(self):
            self.model = model
            self.conf = confidence

        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")

            # Run YOLOv8 prediction
            results = self.model.predict(img, conf=self.conf)

            if results[0].boxes:
                cls_id = int(results[0].boxes.cls[0])
                class_name = self.model.names[cls_id]
                st.session_state.detected_class = class_name
            annotated_frame = results[0].plot()

            return annotated_frame
    webrtc_streamer(
        key="yolo-webcam",
        video_processor_factory=YOLOTransformer,
        media_stream_constraints={"video": True, "audio": False},
    )

with savedVideo:
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"], key = "video upload")
    if uploaded_file:
        # Save uploaded video temporarily
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

        save_dir = tempfile.mkdtemp()

        results = model(temp_file_path, conf=confidence, save=True, project=save_dir, name="detect")

         # YOLO saves output inside save_dir/detect/
        output_path = os.path.join(save_dir, "detect", os.path.basename(temp_file_path))

        # Display the annotated video
        st.video(output_path)

# --- Main thread: handle audio playback ---
if st.session_state.detected_class:
    
    # avoid spamming same sound too fast
    if st.session_state.detected_class != st.session_state.last_played:
        print("playing audio")
        play_audio(st.session_state.detected_class)
        st.session_state.last_played = st.session_state.detected_class