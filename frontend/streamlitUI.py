import streamlit as st

ALLOWED_USERS = {
    "laliparth@gmail.com",
    "zaiddyd42@gmail.com",
    "shreyagoyal1733@gmail.com",
    "dharneet08@gmail.com",
    "shounak6425@gmail.com",
    "guptatanush763@gmail.com"
}


def login_screen():
    st.title("Intelligent Border Video Analytics Platform")
    st.subheader("Secure Surveillance Portal")

    st.button("Login with Google", on_click=st.login)


if not st.user.is_logged_in:
    login_screen()
    st.stop()


if st.user.email not in ALLOWED_USERS:
    st.error("You are not authorized to access this system.")
    st.button("Logout", on_click=st.logout)
    st.stop()
    


import cv2
import time
import re
import os
from difflib import SequenceMatcher
import pandas as pd
from datetime import datetime
from ultralytics import YOLO
import easyocr

# ============================================================
# IBVAP - INTELLIGENT BORDER VIDEO ANALYTICS PLATFORM
# ============================================================

st.set_page_config(
    page_title="IBVAP",
    page_icon="🛡️",
    layout="wide"
)


# ============================================================
# CONFIGURATION
# ============================================================

VIDEO_FOLDER = "/mount/src/sih-repository/frontend/demo_videos"

VEHICLE_MODEL = "SIH-Repository/frontend/yolo26n.pt"
PLATE_MODEL = "SIH-Repository/frontend/models/plate_detector.pt"

VEHICLE_EVERY = 2
PLATE_EVERY = 5
OCR_EVERY = 15


# ============================================================
# HEADER
# ============================================================

st.title("🛡️ IBVAP")

st.subheader(
    "Intelligent Border Video Analytics Platform"
)

st.caption(
    "AI-powered surveillance using existing CCTV infrastructure"
)

st.success(f"Welcome, {st.user.name}!")

if st.button("Logout"):
    st.logout()

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🛡️ IBVAP")

st.sidebar.caption("Surveillance Control Panel")

mode = st.sidebar.radio(
    "Detection Mode",
    [
        "🔢 License Plate Detection",
        "👤 Person Detection",
        "📊 System Overview"
    ]
)

st.sidebar.divider()

st.sidebar.subheader("🎥 Surveillance Footage")

if not os.path.exists(VIDEO_FOLDER):
    st.error("demo_videos folder not found.")
    st.stop()

video_files = [
    f for f in os.listdir(VIDEO_FOLDER)
    if f.lower().endswith(
        (".mp4", ".mov", ".avi", ".mkv")
    )
]

if not video_files:
    st.error("No videos found in demo_videos folder.")
    st.stop()

selected_video = st.sidebar.selectbox(
    "Select Video",
    video_files
)

VIDEO_PATH = os.path.join(
    VIDEO_FOLDER,
    selected_video
)

st.sidebar.divider()

st.sidebar.info(
    "Select a detection mode and start surveillance."
)


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    vehicle_model = YOLO(
        VEHICLE_MODEL
    )

    plate_model = YOLO(
        PLATE_MODEL
    )

    reader = easyocr.Reader(
        ["en"],
        gpu=False
    )

    return (
        vehicle_model,
        plate_model,
        reader
    )


vehicle_model, plate_model, reader = load_models()


# ============================================================
# OCR FUNCTION
# ============================================================

def read_plate(image):

    if image is None or image.size == 0:
        return ""

    try:

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        # Upscale small plates
        gray = cv2.resize(
            gray,
            None,
            fx=3,
            fy=3,
            interpolation=cv2.INTER_CUBIC
        )

        # Improve contrast
        gray = cv2.equalizeHist(gray)

        results = reader.readtext(
            gray,
            detail=1,
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
            mag_ratio=1
        )

        if not results:
            return ""

        best = max(
            results,
            key=lambda x: x[2]
        )

        text = best[1]

        text = re.sub(
            r"[^A-Z0-9]",
            "",
            text.upper()
        )

        if len(text) < 3:
            return ""

        return text

    except Exception:
        return ""


# ============================================================
# PLATE MATCHING / DEDUPLICATION
# ============================================================

def plate_similarity(a, b):
    """Return similarity between two OCR plate readings."""
    a = re.sub(r"[^A-Z0-9]", "", a.upper())
    b = re.sub(r"[^A-Z0-9]", "", b.upper())

    if not a or not b:
        return 0.0

    # Exact match
    if a == b:
        return 1.0

    # One reading is a partial version of the other.
    if len(a) >= 4 and len(b) >= 4:
        if a in b or b in a:
            return 0.90

    return SequenceMatcher(None, a, b).ratio()


def find_matching_plate(text, plate_data, threshold=0.75):
    """
    Find an existing plate identity for a new OCR reading.

    This prevents small OCR variations of the same vehicle
    from being counted as separate unique plates.
    """
    best_match = None
    best_score = 0.0

    for existing in plate_data.keys():
        score = plate_similarity(text, existing)

        if score > best_score:
            best_score = score
            best_match = existing

    if best_score >= threshold:
        return best_match

    return None


def update_plate_record(text, plate_data, now):
    """
    Add a new plate or merge it into an existing similar reading.
    Returns the canonical plate key.
    """
    matched = find_matching_plate(
        text,
        plate_data,
        threshold=0.75
    )

    if matched is None:
        plate_data[text] = {
            "Number Plate": text,
            "First Seen": now,
            "Last Seen": now,
            "Detections": 1,
            "Status": "Scanned"
        }
        return text

    # Same/very similar plate was already seen.
    plate_data[matched]["Last Seen"] = now
    plate_data[matched]["Detections"] += 1

    return matched


# ============================================================
# VIDEO INFORMATION
# ============================================================

def get_video_info():

    cap = cv2.VideoCapture(
        VIDEO_PATH
    )

    if not cap.isOpened():
        return None

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:
        fps = 25

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    cap.release()

    return (
        fps,
        total_frames,
        width,
        height
    )


# ============================================================
# START BUTTON
# ============================================================

start = st.button(
    "▶ START SURVEILLANCE",
    type="primary",
    use_container_width=True
)


# ============================================================
# LICENSE PLATE MODE
# ============================================================

if start and mode == "🔢 License Plate Detection":

    st.header("🔢 License Plate Detection")

    st.caption(
        "Automatic license plate detection and OCR analysis"
    )

    video_info = get_video_info()

    if video_info is None:
        st.error("Could not open video.")
        st.stop()

    fps, total_frames, width, height = video_info

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    m1, m2, m3, m4 = st.columns(4)

    system_metric = m1.empty()
    plate_metric = m2.empty()
    detection_metric = m3.empty()
    status_metric = m4.empty()

    system_metric.metric(
        "SYSTEM",
        "ONLINE"
    )

    plate_metric.metric(
        "UNIQUE PLATES",
        0
    )

    st.caption(
        "Similar OCR readings are merged into one plate identity."
    )

    detection_metric.metric(
        "DETECTIONS",
        0
    )

    status_metric.metric(
        "STATUS",
        "SCANNING"
    )

    st.divider()

    # --------------------------------------------------------
    # MAIN DISPLAY
    # --------------------------------------------------------

    left, right = st.columns(
        [2.3, 1]
    )

    with left:

        st.subheader(
            "📹 License Surveillance Feed"
        )

        video_placeholder = st.empty()

    with right:

        st.subheader(
            "🔢 Latest Plate"
        )

        latest_plate_placeholder = st.empty()

        st.subheader(
            "📊 Detection Status"
        )

        detection_status_placeholder = st.empty()

        progress_placeholder = st.empty()

    # --------------------------------------------------------
    # DATABASE
    # --------------------------------------------------------

    st.divider()

    st.subheader(
        "🗃️ License Plate Database"
    )

    table_placeholder = st.empty()

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    plate_data = {}

    latest_plate = "Scanning..."

    last_plate_box = None

    frame_number = 0

    total_detections = 0

    # --------------------------------------------------------
    # VIDEO
    # --------------------------------------------------------

    cap = cv2.VideoCapture(
        VIDEO_PATH
    )

    if not cap.isOpened():

        st.error(
            "Could not open surveillance footage."
        )

        st.stop()

    while True:

        success, frame = cap.read()

        if not success:
            break

        frame_number += 1

        # ----------------------------------------------------
        # PLATE DETECTION
        # ----------------------------------------------------

        if frame_number % PLATE_EVERY == 0:

            results = plate_model(
                frame,
                verbose=False
            )

            best_plate = None
            best_confidence = 0

            for result in results:

                for box in result.boxes:

                    confidence = float(
                        box.conf[0]
                    )

                    if confidence < 0.35:
                        continue

                    if confidence > best_confidence:

                        best_confidence = confidence

                        best_plate = tuple(
                            map(
                                int,
                                box.xyxy[0]
                            )
                        )

            if best_plate:

                last_plate_box = best_plate

        # ----------------------------------------------------
        # DRAW PLATE
        # ----------------------------------------------------

        if last_plate_box:

            x1, y1, x2, y2 = last_plate_box

            x1 = max(
                0,
                min(
                    x1,
                    frame.shape[1] - 1
                )
            )

            y1 = max(
                0,
                min(
                    y1,
                    frame.shape[0] - 1
                )
            )

            x2 = max(
                x1 + 1,
                min(
                    x2,
                    frame.shape[1]
                )
            )

            y2 = max(
                y1 + 1,
                min(
                    y2,
                    frame.shape[0]
                )
            )

            # ------------------------------------------------
            # OCR
            # ------------------------------------------------

            if frame_number % OCR_EVERY == 0:

                plate_crop = frame[
                    y1:y2,
                    x1:x2
                ]

                text = read_plate(
                    plate_crop
                )

                if text:

                    latest_plate = text

                    total_detections += 1

                    now = datetime.now().strftime(
                        "%H:%M:%S"
                    )

                    # Merge OCR variations instead of counting
                    # every slightly different reading as a new plate.
                    update_plate_record(
                        text,
                        plate_data,
                        now
                    )

                # --------------------------------------------
                # UPDATE TABLE
                # --------------------------------------------

                if plate_data:

                    df = pd.DataFrame(
                        list(
                            plate_data.values()
                        )
                    )

                    table_placeholder.dataframe(
                        df,
                        use_container_width=True,
                        hide_index=True
                    )

            # ------------------------------------------------
            # DRAW BLUE PLATE BOX
            # ------------------------------------------------

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (255, 0, 0),
                3
            )

            if latest_plate != "Scanning...":

                cv2.putText(
                    frame,
                    f"PLATE: {latest_plate}",
                    (
                        x1,
                        max(y1 - 10, 20)
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 0, 0),
                    2
                )

        # ----------------------------------------------------
        # UI
        # ----------------------------------------------------

        latest_plate_placeholder.metric(
            "Latest Plate",
            latest_plate
        )

        plate_metric.metric(
            "UNIQUE PLATES",
            len(plate_data)
        )

        detection_metric.metric(
            "DETECTIONS",
            total_detections
        )

        if latest_plate != "Scanning...":

            detection_status_placeholder.success(
                "🔍 PLATE DETECTED"
            )

        else:

            detection_status_placeholder.info(
                "🔎 SCANNING..."
            )

        progress = (
            frame_number /
            max(total_frames, 1)
        )

        progress_placeholder.progress(
            min(progress, 1.0),
            text=f"Processing: {int(progress * 100)}%"
        )

        # ----------------------------------------------------
        # DISPLAY
        # ----------------------------------------------------

        frame_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        video_placeholder.image(
            frame_rgb,
            channels="RGB",
            use_container_width=True
        )

        time.sleep(
            max(
                1 / fps,
                0.01
            )
        )

    cap.release()

    st.success(
        "✅ License plate analysis complete."
    )

    # --------------------------------------------------------
    # FINAL REPORT
    # --------------------------------------------------------

    if plate_data:

        st.subheader(
            "📊 Final Detection Report"
        )

        final_df = pd.DataFrame(
            list(
                plate_data.values()
            )
        )

        st.dataframe(
            final_df,
            use_container_width=True,
            hide_index=True
        )

        csv_data = final_df.to_csv(
            index=False
        )

        st.download_button(
            "📥 Download CSV",
            data=csv_data,
            file_name="IBVAP_license_database.csv",
            mime="text/csv"
        )


# ============================================================
# PERSON DETECTION MODE
# ============================================================

elif start and mode == "👤 Person Detection":

    st.header("👤 Person Detection")

    st.caption(
        "Real-time AI-based human detection from surveillance footage"
    )

    video_info = get_video_info()

    if video_info is None:
        st.error("Could not open video.")
        st.stop()

    fps, total_frames, width, height = video_info

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    m1, m2, m3, m4 = st.columns(4)

    system_metric = m1.empty()
    people_metric = m2.empty()
    peak_metric = m3.empty()
    status_metric = m4.empty()

    system_metric.metric(
        "SYSTEM",
        "ONLINE"
    )

    people_metric.metric(
        "PEOPLE",
        0
    )

    peak_metric.metric(
        "PEAK COUNT",
        0
    )

    status_metric.metric(
        "STATUS",
        "SCANNING"
    )

    st.divider()

    left, right = st.columns(
        [2.3, 1]
    )

    with left:

        st.subheader(
            "📹 Person Surveillance Feed"
        )

        video_placeholder = st.empty()

    with right:

        st.subheader(
            "👤 Live People"
        )

        people_placeholder = st.empty()

        st.subheader(
            "🚨 Security Status"
        )

        security_placeholder = st.empty()

        progress_placeholder = st.empty()

    st.divider()

    st.subheader(
        "📋 Person Detection Log"
    )

    event_placeholder = st.empty()

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    frame_number = 0

    people = 0

    peak_people = 0

    events = []

    last_count = -1

    last_event_time = 0

    last_boxes = []

    # --------------------------------------------------------
    # VIDEO
    # --------------------------------------------------------

    cap = cv2.VideoCapture(
        VIDEO_PATH
    )

    if not cap.isOpened():

        st.error(
            "Could not open surveillance footage."
        )

        st.stop()

    while True:

        success, frame = cap.read()

        if not success:
            break

        frame_number += 1

        # ----------------------------------------------------
        # PERSON DETECTION
        # ----------------------------------------------------

        if frame_number % VEHICLE_EVERY == 0:

            results = vehicle_model(
                frame,
                verbose=False
            )

            detected_people = []

            for result in results:

                for box in result.boxes:

                    confidence = float(
                        box.conf[0]
                    )

                    if confidence < 0.40:
                        continue

                    class_id = int(
                        box.cls[0]
                    )

                    class_name = (
                        vehicle_model.names[
                            class_id
                        ]
                    )

                    if class_name != "person":
                        continue

                    x1, y1, x2, y2 = map(
                        int,
                        box.xyxy[0]
                    )

                    detected_people.append(
                        (
                            x1,
                            y1,
                            x2,
                            y2,
                            confidence
                        )
                    )

            last_boxes = detected_people

            people = len(
                detected_people
            )

            peak_people = max(
                peak_people,
                people
            )

        # ----------------------------------------------------
        # DRAW PERSON BOXES
        # ----------------------------------------------------

        for (
            x1,
            y1,
            x2,
            y2,
            confidence
        ) in last_boxes:

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 0, 255),
                2
            )

            cv2.putText(
                frame,
                f"PERSON {confidence:.2f}",
                (
                    x1,
                    max(y1 - 8, 20)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2
            )

        # ----------------------------------------------------
        # EVENTS
        # ----------------------------------------------------

        current_time = time.time()

        if (
            people != last_count
            and current_time - last_event_time > 1
        ):

            timestamp = datetime.now().strftime(
                "%H:%M:%S"
            )

            if people > 0:

                events.insert(
                    0,
                    f"{timestamp}  👤 People detected: {people}"
                )

            elif last_count > 0:

                events.insert(
                    0,
                    f"{timestamp}  ✅ Area clear"
                )

            events = events[:8]

            last_count = people

            last_event_time = current_time

        # ----------------------------------------------------
        # UI
        # ----------------------------------------------------

        people_metric.metric(
            "PEOPLE",
            people
        )

        peak_metric.metric(
            "PEAK COUNT",
            peak_people
        )

        if people > 0:

            status_metric.metric(
                "STATUS",
                "ACTIVE"
            )

            security_placeholder.warning(
                f"⚠️ {people} PERSON(S) DETECTED"
            )

        else:

            status_metric.metric(
                "STATUS",
                "CLEAR"
            )

            security_placeholder.success(
                "🟢 AREA CLEAR"
            )

        people_placeholder.metric(
            "Current People",
            people
        )

        progress = (
            frame_number /
            max(total_frames, 1)
        )

        progress_placeholder.progress(
            min(progress, 1.0),
            text=f"Processing: {int(progress * 100)}%"
        )

        # ----------------------------------------------------
        # EVENT LOG
        # ----------------------------------------------------

        if events:

            event_placeholder.code(
                "\n\n".join(events)
            )

        else:

            event_placeholder.info(
                "Waiting for detection events..."
            )

        # ----------------------------------------------------
        # DISPLAY
        # ----------------------------------------------------

        frame_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        video_placeholder.image(
            frame_rgb,
            channels="RGB",
            use_container_width=True
        )

        time.sleep(
            max(
                1 / fps,
                0.01
            )
        )

    cap.release()

    st.success(
        "✅ Person detection analysis complete."
    )


# ============================================================
# SYSTEM OVERVIEW
# ============================================================

elif start and mode == "📊 System Overview":

    st.header("📊 System Overview")

    st.caption(
        "High-level surveillance system monitoring"
    )

    video_info = get_video_info()

    if video_info is None:
        st.error("Could not open video.")
        st.stop()

    fps, total_frames, width, height = video_info

    # --------------------------------------------------------
    # INTRO
    # --------------------------------------------------------

    st.info(
        "IBVAP combines vehicle detection, human detection "
        "and license plate recognition into a single "
        "surveillance platform."
    )

    st.divider()

    # --------------------------------------------------------
    # SYSTEM COMPONENTS
    # --------------------------------------------------------

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "🚗 Vehicle AI",
        "ONLINE"
    )

    c2.metric(
        "👤 Person AI",
        "ONLINE"
    )

    c3.metric(
        "🔢 ANPR",
        "ONLINE"
    )

    st.divider()

    # --------------------------------------------------------
    # VIDEO INFO
    # --------------------------------------------------------

    st.subheader(
        "🎥 Current Surveillance Feed"
    )

    info1, info2, info3, info4 = st.columns(4)

    info1.metric(
        "Video",
        selected_video
    )

    info2.metric(
        "Resolution",
        f"{width} × {height}"
    )

    info3.metric(
        "FPS",
        f"{fps:.1f}"
    )

    info4.metric(
        "Frames",
        total_frames
    )

    st.divider()

    # --------------------------------------------------------
    # CAPABILITIES
    # --------------------------------------------------------

    st.subheader(
        "🧠 AI Capabilities"
    )

    cap1, cap2 = st.columns(2)

    with cap1:

        st.success(
            "🔢 License Plate Recognition"
        )

        st.write(
            "Detect vehicles' license plates and "
            "extract readable characters using OCR."
        )

        st.success(
            "👤 Person Detection"
        )

        st.write(
            "Detect people in surveillance footage "
            "and monitor changes in activity."
        )

    with cap2:

        st.success(
            "🚗 Vehicle Detection"
        )

        st.write(
            "Detect cars, trucks, buses and motorcycles "
            "using YOLO."
        )

        st.success(
            "📊 Detection Database"
        )

        st.write(
            "Store detected license plates with "
            "timestamps and detection counts."
        )

    st.divider()

    st.subheader(
        "🚀 IBVAP Architecture"
    )

    st.code(
        """
Camera / Video
       ↓
YOLO Detection
       ↓
┌───────────────┬───────────────┐
│               │               │
Vehicle       Person          Plate
Detection     Detection       Detection
│               │               │
└───────────────┴───────────────┘
                ↓
              OCR
                ↓
        Detection Database
                ↓
        Streamlit Dashboard
        """,
        language="text"
    )

else:

    # --------------------------------------------------------
    # LANDING PAGE
    # --------------------------------------------------------

    st.header(
        "Welcome to IBVAP"
    )

    st.write(
        "Select a detection mode from the sidebar "
        "and press **START SURVEILLANCE**."
    )

    st.divider()

    c1, c2, c3 = st.columns(3)

    with c1:

        st.info(
            "### 🔢 License Plates\n"
            "Detect and read vehicle license plates."
        )

    with c2:

        st.info(
            "### 👤 People\n"
            "Detect people in surveillance footage."
        )

    with c3:

        st.info(
            "### 📊 Overview\n"
            "View the complete IBVAP architecture."
        )

