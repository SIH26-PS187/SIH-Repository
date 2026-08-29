import cv2
import os
import re
import csv
import math
from collections import Counter, defaultdict
from difflib import SequenceMatcher

from ultralytics import YOLO
import easyocr


# ============================================================
# IBVAP ANPR V6
# ============================================================

PLATE_MODEL = "models/plate_detector.pt"

VIDEO_PATH = "demo_videos/indian.mp4"

OUTPUT_PATH = "results/indian_anpr_v6.mp4"
CSV_PATH = "results/plate_database_v6.csv"

# Speed
FRAME_SKIP = 2
OCR_INTERVAL = 4

# Detection / OCR
PLATE_CONF = 0.15
OCR_CONF = 0.20

# History
MAX_HISTORY = 25

# Minimum plate size
MIN_PLATE_WIDTH = 20
MIN_PLATE_HEIGHT = 7

# Quality thresholds
MIN_SHARPNESS = 18
MIN_BRIGHTNESS = 25
MAX_BRIGHTNESS = 245

# Tracking
TRACK_IOU_THRESHOLD = 0.20
TRACK_MAX_AGE = 60

# Final confidence
VERIFIED_CONF = 0.80
PARTIAL_CONF = 0.55


# ============================================================
# LOAD MODELS
# ============================================================

print("Loading license plate model...")

plate_model = YOLO(PLATE_MODEL)

print("Loading OCR...")

reader = easyocr.Reader(
    ["en"],
    gpu=False,
    verbose=False
)

print()
print("Models loaded successfully.")
print()


# ============================================================
# CLEAN TEXT
# ============================================================

def clean_text(text):

    text = str(text).upper()

    text = re.sub(
        r"[^A-Z0-9]",
        "",
        text
    )

    return text


# ============================================================
# BASIC PLATE CHECK
# ============================================================

def possible_plate(text):

    text = clean_text(text)

    if len(text) < 3:
        return False

    if len(text) > 12:
        return False

    if len(set(text)) == 1:
        return False

    return True


# ============================================================
# PLATE QUALITY
# ============================================================

def plate_quality(crop):

    if crop is None or crop.size == 0:
        return 0.0

    h, w = crop.shape[:2]

    if w < MIN_PLATE_WIDTH or h < MIN_PLATE_HEIGHT:
        return 0.0

    gray = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2GRAY
    )

    # --------------------------------------------------------
    # Sharpness
    # --------------------------------------------------------

    sharpness = cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()

    sharp_score = min(
        sharpness / 120.0,
        1.0
    )

    # --------------------------------------------------------
    # Brightness
    # --------------------------------------------------------

    brightness = float(gray.mean())

    if (
        brightness < MIN_BRIGHTNESS
        or brightness > MAX_BRIGHTNESS
    ):
        brightness_score = 0.35
    else:
        brightness_score = 1.0

    # --------------------------------------------------------
    # Resolution
    # --------------------------------------------------------

    resolution_score = min(
        (w * h) / 5000.0,
        1.0
    )

    # --------------------------------------------------------
    # Aspect ratio
    # --------------------------------------------------------

    ratio = w / max(h, 1)

    if 1.5 <= ratio <= 7.0:
        ratio_score = 1.0
    elif 1.2 <= ratio <= 8.0:
        ratio_score = 0.7
    else:
        ratio_score = 0.35

    quality = (
        sharp_score * 0.40
        +
        brightness_score * 0.20
        +
        resolution_score * 0.25
        +
        ratio_score * 0.15
    )

    return max(
        0.0,
        min(
            1.0,
            quality
        )
    )


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def make_variants(crop):

    if crop is None or crop.size == 0:
        return []

    # Upscale
    enlarged = cv2.resize(
        crop,
        None,
        fx=4,
        fy=4,
        interpolation=cv2.INTER_CUBIC
    )

    gray = cv2.cvtColor(
        enlarged,
        cv2.COLOR_BGR2GRAY
    )

    # CLAHE
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(gray)

    # Mild denoise
    denoised = cv2.bilateralFilter(
        enhanced,
        5,
        50,
        50
    )

    # Sharpen
    blur = cv2.GaussianBlur(
        denoised,
        (0, 0),
        1.2
    )

    sharpened = cv2.addWeighted(
        denoised,
        1.5,
        blur,
        -0.5,
        0
    )

    # OTSU
    _, otsu = cv2.threshold(
        sharpened,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Adaptive threshold
    adaptive = cv2.adaptiveThreshold(
        sharpened,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        9
    )

    return [
        enlarged,
        enhanced,
        sharpened,
        otsu,
        adaptive
    ]


# ============================================================
# OCR
# ============================================================

def read_plate(crop):

    quality = plate_quality(crop)

    if quality <= 0:
        return "", 0.0, 0.0

    variants = make_variants(crop)

    candidates = []

    for image in variants:

        try:

            results = reader.readtext(
                image,
                detail=1,
                paragraph=False,
                allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
            )

            for result in results:

                if len(result) < 3:
                    continue

                text = clean_text(
                    result[1]
                )

                confidence = float(
                    result[2]
                )

                if not possible_plate(text):
                    continue

                if confidence < OCR_CONF:
                    continue

                candidates.append(
                    (
                        text,
                        confidence
                    )
                )

        except Exception:
            continue

    if not candidates:
        return "", 0.0, quality

    # --------------------------------------------------------
    # Group identical readings
    # --------------------------------------------------------

    grouped = defaultdict(list)

    for text, confidence in candidates:
        grouped[text].append(
            confidence
        )

    scored = []

    for text, values in grouped.items():

        avg_conf = (
            sum(values)
            /
            len(values)
        )

        repetitions = len(values)

        length_bonus = (
            0.10
            if 7 <= len(text) <= 10
            else 0.0
        )

        repetition_bonus = min(
            repetitions * 0.04,
            0.16
        )

        score = (
            avg_conf * 0.60
            +
            quality * 0.25
            +
            length_bonus
            +
            repetition_bonus
        )

        scored.append(
            (
                text,
                avg_conf,
                score
            )
        )

    scored.sort(
        key=lambda x: x[2],
        reverse=True
    )

    best = scored[0]

    return (
        best[0],
        best[1],
        quality
    )


# ============================================================
# STRING SIMILARITY
# ============================================================

def similarity(a, b):

    return SequenceMatcher(
        None,
        a,
        b
    ).ratio()


# ============================================================
# ALIGN STRINGS
# ============================================================

def align_strings(a, b):

    n = len(a)
    m = len(b)

    dp = [
        [0] * (m + 1)
        for _ in range(n + 1)
    ]

    for i in range(n + 1):
        dp[i][0] = i

    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):

        for j in range(1, m + 1):

            cost = (
                0
                if a[i - 1] == b[j - 1]
                else 1
            )

            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost
            )

    i = n
    j = m

    aligned_a = []
    aligned_b = []

    while i > 0 or j > 0:

        if (
            i > 0
            and j > 0
            and dp[i][j]
            ==
            dp[i - 1][j - 1]
            +
            (
                0
                if a[i - 1] == b[j - 1]
                else 1
            )
        ):

            aligned_a.append(
                a[i - 1]
            )

            aligned_b.append(
                b[j - 1]
            )

            i -= 1
            j -= 1

        elif (
            i > 0
            and
            dp[i][j]
            ==
            dp[i - 1][j] + 1
        ):

            aligned_a.append(
                a[i - 1]
            )

            aligned_b.append("?")

            i -= 1

        else:

            aligned_a.append("?")

            aligned_b.append(
                b[j - 1]
            )

            j -= 1

    aligned_a.reverse()
    aligned_b.reverse()

    return (
        "".join(aligned_a),
        "".join(aligned_b)
    )


# ============================================================
# BUILD CONSENSUS V6
# ============================================================

def build_consensus(history):

    if not history:
        return "", 0.0, "LOW CONFIDENCE"

    observations = []

    for item in history:

        text = item["text"]

        if not possible_plate(text):
            continue

        observations.append(item)

    if not observations:
        return "", 0.0, "LOW CONFIDENCE"

    # --------------------------------------------------------
    # Find strongest anchor
    # --------------------------------------------------------

    anchor_item = max(
        observations,
        key=lambda x:
            (
                x["confidence"]
                *
                x["quality"],
                len(x["text"])
            )
    )

    anchor = anchor_item["text"]

    # --------------------------------------------------------
    # Cluster similar observations
    # --------------------------------------------------------

    cluster = []

    for item in observations:

        text = item["text"]

        sim = similarity(
            anchor,
            text
        )

        if (
            sim >= 0.45
            or text in anchor
            or anchor in text
        ):

            cluster.append(
                item
            )

    if not cluster:
        cluster = [
            anchor_item
        ]

    # --------------------------------------------------------
    # Position votes
    # --------------------------------------------------------

    votes = [
        Counter()
        for _ in range(len(anchor))
    ]

    weighted_votes = [
        defaultdict(float)
        for _ in range(len(anchor))
    ]

    position_conf = [
        []
        for _ in range(len(anchor))
    ]

    for item in cluster:

        text = item["text"]

        weight = (
            item["confidence"]
            *
            item["quality"]
        )

        aligned_anchor, aligned_text = (
            align_strings(
                anchor,
                text
            )
        )

        anchor_index = 0

        for a_char, b_char in zip(
            aligned_anchor,
            aligned_text
        ):

            if a_char == "?":
                continue

            if anchor_index >= len(anchor):
                break

            if b_char != "?":

                votes[
                    anchor_index
                ][b_char] += 1

                weighted_votes[
                    anchor_index
                ][b_char] += weight

                position_conf[
                    anchor_index
                ].append(
                    weight
                )

            anchor_index += 1

    # --------------------------------------------------------
    # Build result
    # --------------------------------------------------------

    result = []

    character_scores = []

    for i in range(len(anchor)):

        if not votes[i]:

            result.append("?")
            character_scores.append(0.0)

            continue

        character = max(
            weighted_votes[i],
            key=weighted_votes[i].get
        )

        weighted_total = sum(
            weighted_votes[i].values()
        )

        selected_weight = (
            weighted_votes[i][character]
        )

        agreement = (
            selected_weight
            /
            max(
                weighted_total,
                1e-6
            )
        )

        avg_weight = (
            sum(
                position_conf[i]
            )
            /
            max(
                len(position_conf[i]),
                1
            )
        )

        # Require meaningful agreement
        if agreement >= 0.52:

            result.append(
                character
            )

            character_scores.append(
                min(
                    1.0,
                    avg_weight
                    *
                    agreement
                )
            )

        else:

            result.append("?")
            character_scores.append(0.0)

    final_text = "".join(result)

    # --------------------------------------------------------
    # Character confidence
    # --------------------------------------------------------

    known_scores = [
        x
        for x in character_scores
        if x > 0
    ]

    if known_scores:

        character_conf = (
            sum(known_scores)
            /
            len(known_scores)
        )

    else:

        character_conf = 0.0

    # --------------------------------------------------------
    # Temporal consistency
    # --------------------------------------------------------

    unique_texts = Counter(
        item["text"]
        for item in cluster
    )

    strongest_count = (
        unique_texts.most_common(1)[0][1]
        if unique_texts
        else 0
    )

    temporal_consistency = min(
        strongest_count / 5.0,
        1.0
    )

    # More observations = more confidence,
    # but with a cap.
    observation_bonus = min(
        len(cluster) / 20.0,
        0.15
    )

    final_confidence = (
        character_conf * 0.60
        +
        temporal_consistency * 0.25
        +
        observation_bonus
    )

    final_confidence = max(
        0.0,
        min(
            1.0,
            final_confidence
        )
    )

    unknown_count = final_text.count("?")

    # --------------------------------------------------------
    # Status
    # --------------------------------------------------------

    if (
        final_confidence >= VERIFIED_CONF
        and unknown_count == 0
        and len(final_text) >= 7
        and len(cluster) >= 3
    ):

        status = "VERIFIED"

    elif (
        final_confidence >= PARTIAL_CONF
        and len(final_text) >= 4
    ):

        status = "PARTIAL"

    else:

        status = "LOW CONFIDENCE"

    return (
        final_text,
        final_confidence,
        status
    )


# ============================================================
# IOU
# ============================================================

def iou(box1, box2):

    x1 = max(
        box1[0],
        box2[0]
    )

    y1 = max(
        box1[1],
        box2[1]
    )

    x2 = min(
        box1[2],
        box2[2]
    )

    y2 = min(
        box1[3],
        box2[3]
    )

    intersection = (
        max(
            0,
            x2 - x1
        )
        *
        max(
            0,
            y2 - y1
        )
    )

    area1 = (
        max(
            0,
            box1[2] - box1[0]
        )
        *
        max(
            0,
            box1[3] - box1[1]
        )
    )

    area2 = (
        max(
            0,
            box2[2] - box2[0]
        )
        *
        max(
            0,
            box2[3] - box2[1]
        )
    )

    union = (
        area1
        +
        area2
        -
        intersection
    )

    if union <= 0:
        return 0.0

    return intersection / union


# ============================================================
# TRACK MANAGEMENT
# ============================================================

tracks = {}

next_track_id = 0


def get_track(box, frame_number):

    global next_track_id

    best_id = None
    best_overlap = 0.0

    for track_id, track in tracks.items():

        overlap = iou(
            box,
            track["box"]
        )

        if overlap > best_overlap:

            best_overlap = overlap
            best_id = track_id

    if (
        best_id is not None
        and
        best_overlap >= TRACK_IOU_THRESHOLD
    ):

        tracks[
            best_id
        ]["box"] = box

        tracks[
            best_id
        ]["last_seen"] = frame_number

        return best_id

    # New track

    track_id = next_track_id

    next_track_id += 1

    tracks[track_id] = {

        "box": box,

        "last_seen": frame_number,

        "last_ocr_frame": -999,

        "history": [],

        "plate": "",

        "confidence": 0.0,

        "status": "SCANNING"

    }

    return track_id


# ============================================================
# MAIN
# ============================================================

def process_video():

    global tracks

    if not os.path.exists(
        VIDEO_PATH
    ):

        print(
            f"ERROR: Video not found: "
            f"{VIDEO_PATH}"
        )

        return

    os.makedirs(
        "results",
        exist_ok=True
    )

    cap = cv2.VideoCapture(
        VIDEO_PATH
    )

    if not cap.isOpened():

        print(
            "ERROR: Could not open video."
        )

        return

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

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    if fps <= 0:
        fps = 30

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        OUTPUT_PATH,
        fourcc,
        fps,
        (width, height)
    )

    detected_plates = {}

    frame_number = 0

    print(
        "===================================="
    )

    print(
        "IBVAP ANPR V6"
    )

    print(
        "===================================="
    )

    print(
        f"Video: {VIDEO_PATH}"
    )

    print(
        f"Resolution: {width} x {height}"
    )

    print(
        f"FPS: {fps}"
    )

    print(
        f"Frames: {total_frames}"
    )

    print(
        "===================================="
    )

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_number += 1

        # ----------------------------------------------------
        # Frame skipping
        # ----------------------------------------------------

        if (
            frame_number % FRAME_SKIP
            != 0
        ):

            writer.write(frame)

            continue

        # ----------------------------------------------------
        # Detection
        # ----------------------------------------------------

        results = plate_model(
            frame,
            conf=PLATE_CONF,
            verbose=False
        )

        if results:

            result = results[0]

            if result.boxes is not None:

                for box in result.boxes:

                    detector_confidence = float(
                        box.conf[0]
                    )

                    coords = (
                        box.xyxy[0]
                        .cpu()
                        .numpy()
                        .astype(int)
                    )

                    x1, y1, x2, y2 = coords

                    x1 = max(
                        0,
                        x1
                    )

                    y1 = max(
                        0,
                        y1
                    )

                    x2 = min(
                        width,
                        x2
                    )

                    y2 = min(
                        height,
                        y2
                    )

                    plate_width = (
                        x2 - x1
                    )

                    plate_height = (
                        y2 - y1
                    )

                    if (
                        plate_width
                        <
                        MIN_PLATE_WIDTH
                        or
                        plate_height
                        <
                        MIN_PLATE_HEIGHT
                    ):

                        continue

                    # ------------------------------------------------
                    # Slightly larger crop
                    # ------------------------------------------------

                    px = int(
                        plate_width * 0.15
                    )

                    py = int(
                        plate_height * 0.30
                    )

                    cx1 = max(
                        0,
                        x1 - px
                    )

                    cy1 = max(
                        0,
                        y1 - py
                    )

                    cx2 = min(
                        width,
                        x2 + px
                    )

                    cy2 = min(
                        height,
                        y2 + py
                    )

                    crop = frame[
                        cy1:cy2,
                        cx1:cx2
                    ]

                    if crop.size == 0:
                        continue

                    # ------------------------------------------------
                    # Track
                    # ------------------------------------------------

                    track_id = get_track(
                        [
                            x1,
                            y1,
                            x2,
                            y2
                        ],
                        frame_number
                    )

                    track = tracks[
                        track_id
                    ]

                    # ------------------------------------------------
                    # OCR
                    # ------------------------------------------------

                    if (
                        frame_number
                        -
                        track["last_ocr_frame"]
                        >= OCR_INTERVAL
                    ):

                        (
                            text,
                            ocr_confidence,
                            quality
                        ) = read_plate(
                            crop
                        )

                        track[
                            "last_ocr_frame"
                        ] = frame_number

                        if text:

                            track[
                                "history"
                            ].append({

                                "text": text,

                                "confidence":
                                    ocr_confidence,

                                "quality":
                                    quality,

                                "detector":
                                    detector_confidence

                            })

                            track[
                                "history"
                            ] = track[
                                "history"
                            ][
                                -MAX_HISTORY:
                            ]

                    # ------------------------------------------------
                    # Consensus
                    # ------------------------------------------------

                    (
                        plate,
                        confidence,
                        status
                    ) = build_consensus(
                        track["history"]
                    )

                    if plate:

                        track[
                            "plate"
                        ] = plate

                        track[
                            "confidence"
                        ] = confidence

                        track[
                            "status"
                        ] = status

                        # Save useful readings
                        if status != "LOW CONFIDENCE":

                            detected_plates[
                                plate
                            ] = {

                                "confidence":
                                    confidence,

                                "status":
                                    status,

                                "track":
                                    track_id

                            }

                    # ------------------------------------------------
                    # Box
                    # ------------------------------------------------

                    if status == "VERIFIED":

                        box_color = (
                            0,
                            255,
                            0
                        )

                    elif status == "PARTIAL":

                        box_color = (
                            0,
                            200,
                            255
                        )

                    else:

                        box_color = (
                            0,
                            165,
                            255
                        )

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        box_color,
                        2
                    )

                    # ------------------------------------------------
                    # Label
                    # ------------------------------------------------

                    if plate:

                        label = (
                            f"{plate} "
                            f"{confidence:.0%}"
                            f" | {status}"
                        )

                    else:

                        label = (
                            "SCANNING..."
                        )

                    label_y = max(
                        25,
                        y1 - 8
                    )

                    cv2.putText(
                        frame,
                        label,
                        (
                            x1,
                            label_y
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.50,
                        box_color,
                        2
                    )

        # ----------------------------------------------------
        # Remove expired tracks
        # ----------------------------------------------------

        expired = []

        for track_id, track in tracks.items():

            if (
                frame_number
                -
                track["last_seen"]
                >
                TRACK_MAX_AGE
            ):

                expired.append(
                    track_id
                )

        for track_id in expired:

            del tracks[
                track_id
            ]

        # ----------------------------------------------------
        # Header
        # ----------------------------------------------------

        cv2.putText(
            frame,
            "IBVAP ANPR V6",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Plates: {len(detected_plates)}",
            (15, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (0, 255, 0),
            2
        )

        writer.write(
            frame
        )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if frame_number % 30 == 0:

            progress = (
                frame_number
                /
                max(
                    1,
                    total_frames
                )
            ) * 100

            print(
                f"Processing: "
                f"{progress:.1f}% | "
                f"Plates: "
                f"{len(detected_plates)}"
            )

    # ========================================================
    # CLEANUP
    # ========================================================

    cap.release()
    writer.release()

    # ========================================================
    # SAVE CSV
    # ========================================================

    with open(
        CSV_PATH,
        "w",
        newline=""
    ) as file:

        writer_csv = csv.writer(
            file
        )

        writer_csv.writerow([
            "Number Plate",
            "Confidence",
            "Status",
            "Track ID"
        ])

        for plate, data in (
            detected_plates.items()
        ):

            writer_csv.writerow([

                plate,

                f"{data['confidence']:.2f}",

                data["status"],

                data["track"]

            ])

    # ========================================================
    # FINAL
    # ========================================================

    print()

    print(
        "===================================="
    )

    print(
        "V6 PROCESSING COMPLETE"
    )

    print(
        "===================================="
    )

    print(
        f"Plates detected: "
        f"{len(detected_plates)}"
    )

    print()

    for plate, data in (
        detected_plates.items()
    ):

        print(
            f"{plate} | "
            f"{data['confidence']:.0%} | "
            f"{data['status']}"
        )

    print()

    print(
        f"Video: {OUTPUT_PATH}"
    )

    print(
        f"CSV: {CSV_PATH}"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    process_video()