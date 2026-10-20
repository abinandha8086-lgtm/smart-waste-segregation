from pathlib import Path
import time

import cv2
import serial
import torch
from ultralytics import YOLO


# ============================================================
# SETTINGS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

MODEL_PATH = PROJECT_DIR / "models" / "best.pt"

SERIAL_PORT = "COM3"
BAUD_RATE = 115200

CAMERA_INDEX = 0

# CUDA GPU: 0 means first NVIDIA GPU.
CUDA_DEVICE = 0

# Detection confidence threshold.
CONFIDENCE = 0.35

# Detection must remain stable for this many seconds.
CONFIRMATION_SECONDS = 3.0

# Bounding boxes must overlap enough to be considered same object.
IOU_THRESHOLD = 0.30

# Time to wait while servo performs sorting movement.
SORT_DELAY = 2.0

# Prevent instant repeat sorting of same object.
REMOVAL_CONFIRMATION_SECONDS = 1.0

# Webcam capture size. This does not change YOLO inference size.
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720


# ============================================================
# CUDA CHECK
# ============================================================

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA GPU is not available.\n"
        "Install CUDA-enabled PyTorch and confirm that:\n"
        "torch.cuda.is_available() returns True."
    )

GPU_NAME = torch.cuda.get_device_name(CUDA_DEVICE)

print("=" * 65)
print("SMART WASTE SORTING SYSTEM - CUDA MODE")
print("=" * 65)
print(f"PyTorch version : {torch.__version__}")
print(f"CUDA version    : {torch.version.cuda}")
print(f"GPU             : {GPU_NAME}")
print(f"Model path      : {MODEL_PATH}")
print("=" * 65)

torch.backends.cudnn.benchmark = True


# ============================================================
# LOAD YOLO MODEL ON CUDA GPU
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"YOLO model not found:\n{MODEL_PATH}"
    )

model = YOLO(str(MODEL_PATH))

print("\nModel loaded successfully.")
print("Model classes:")

for class_id, class_name in model.names.items():
    print(f"{class_id}: {class_name}")

print()


# ============================================================
# SERIAL CONNECTION TO ESP32
# ============================================================

try:
    esp32 = serial.Serial(
        port=SERIAL_PORT,
        baudrate=BAUD_RATE,
        timeout=1
    )

    time.sleep(2)

    esp32.reset_input_buffer()
    esp32.reset_output_buffer()

    print(f"ESP32 connected on {SERIAL_PORT}")

except serial.SerialException as error:
    raise RuntimeError(
        f"Could not open ESP32 serial port {SERIAL_PORT}.\n"
        f"Error: {error}"
    )


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

if not cap.isOpened():
    esp32.close()

    raise RuntimeError(
        f"Could not open camera with index {CAMERA_INDEX}."
    )

print("Camera opened successfully.")
print("Press Q in the camera window to quit.\n")


# ============================================================
# CLASSIFICATION MAP
# ============================================================

def classify_waste(class_name):
    """
    Converts detailed YOLO model labels into ESP32 commands.

    ESP32 commands:
    PAPER
    PLASTIC
    OTHERS
    """

    normalized_name = class_name.lower().strip()

    paper_classes = {
        "paper",
        "cardboard"
    }

    plastic_classes = {
        "plastic",
        "plastic bag",
        "plastic bottle",
        "plastic cup",
        "container for household chemicals"
    }

    organic_classes = {
        "organic",
        "bio",
        "biodegradable",
        "food waste"
    }

    if normalized_name in paper_classes:
        return "PAPER"

    if normalized_name in plastic_classes:
        return "PLASTIC"

    if normalized_name in organic_classes:
        return "OTHERS"

    return "OTHERS"


# ============================================================
# BOUNDING BOX IoU
# ============================================================

def calculate_iou(box_a, box_b):
    """
    Calculates Intersection over Union for two boxes.

    Box format:
    (x1, y1, x2, y2)
    """

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    intersection_x1 = max(ax1, bx1)
    intersection_y1 = max(ay1, by1)

    intersection_x2 = min(ax2, bx2)
    intersection_y2 = min(ay2, by2)

    intersection_width = max(0, intersection_x2 - intersection_x1)
    intersection_height = max(0, intersection_y2 - intersection_y1)

    intersection_area = intersection_width * intersection_height

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union_area = area_a + area_b - intersection_area

    if union_area <= 0:
        return 0.0

    return intersection_area / union_area


# ============================================================
# SEND COMMAND TO ESP32
# ============================================================

def send_command(command):
    command = command.upper().strip()

    print(f"Sending to ESP32: {command}")

    esp32.write((command + "\n").encode("utf-8"))
    esp32.flush()

    time.sleep(0.1)


# ============================================================
# DRAW STATUS PANEL
# ============================================================

def draw_status(
    frame,
    status,
    category="",
    confidence=0.0,
    stable_seconds=0.0,
    color=(0, 255, 0)
):
    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (10, 10),
        (500, 155),
        (0, 0, 0),
        -1
    )

    cv2.addWeighted(
        overlay,
        0.55,
        frame,
        0.45,
        0,
        frame
    )

    cv2.putText(
        frame,
        "SMART WASTE SORTER",
        (20, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Status: {status}",
        (20, 68),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2
    )

    if category:
        cv2.putText(
            frame,
            f"Category: {category}",
            (20, 98),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Confidence: {confidence:.0%}",
            (20, 126),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Confirmed: {stable_seconds:.1f}/{CONFIRMATION_SECONDS:.1f}s",
            (270, 126),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 0),
            1
        )


# ============================================================
# MAIN VARIABLES
# ============================================================

previous_category = None
previous_box = None

confirmation_start_time = None

waiting_for_removal = False
no_detection_start_time = None

sorting = False


# ============================================================
# MAIN LOOP
# ============================================================

try:
    while True:
        success, frame = cap.read()

        if not success:
            print("Camera frame error.")
            break

        current_time = time.time()

        # ----------------------------------------------------
        # CUDA YOLO INFERENCE
        # ----------------------------------------------------

        results = model.predict(
            source=frame,
            conf=CONFIDENCE,
            imgsz=640,
            device=CUDA_DEVICE,
            half=True,
            verbose=False
        )

        best_detection = None

        # ----------------------------------------------------
        # PROCESS ALL DETECTIONS
        # ----------------------------------------------------

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                confidence = float(box.conf[0].item())
                class_id = int(box.cls[0].item())

                class_name = result.names[class_id]
                waste_type = classify_waste(class_name)

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0].tolist()
                )

                current_box = (x1, y1, x2, y2)

                # Bounding box colour by final sorting category.
                if waste_type == "PAPER":
                    box_color = (255, 255, 0)

                elif waste_type == "PLASTIC":
                    box_color = (0, 255, 0)

                else:
                    box_color = (0, 165, 255)

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    box_color,
                    3
                )

                label = f"{class_name}: {confidence:.0%}"

                cv2.putText(
                    frame,
                    label,
                    (x1, max(y1 - 12, 25)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    box_color,
                    2
                )

                category_text = f"Sort as: {waste_type}"

                cv2.putText(
                    frame,
                    category_text,
                    (x1, min(y2 + 25, frame.shape[0] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.60,
                    box_color,
                    2
                )

                # Keep only the highest-confidence detection.
                if (
                    best_detection is None
                    or confidence > best_detection["confidence"]
                ):
                    best_detection = {
                        "class_name": class_name,
                        "category": waste_type,
                        "confidence": confidence,
                        "box": current_box
                    }

        # ----------------------------------------------------
        # WAIT FOR OBJECT REMOVAL AFTER SORTING
        # ----------------------------------------------------

        if waiting_for_removal:
            if best_detection is None:
                if no_detection_start_time is None:
                    no_detection_start_time = current_time

                removal_elapsed = (
                    current_time - no_detection_start_time
                )

                draw_status(
                    frame,
                    "REMOVE OBJECT",
                    color=(0, 165, 255)
                )

                cv2.putText(
                    frame,
                    f"Clear view: {removal_elapsed:.1f}/{REMOVAL_CONFIRMATION_SECONDS:.1f}s",
                    (20, 190),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 165, 255),
                    2
                )

                if removal_elapsed >= REMOVAL_CONFIRMATION_SECONDS:
                    waiting_for_removal = False
                    no_detection_start_time = None

                    previous_category = None
                    previous_box = None
                    confirmation_start_time = None

                    print("Object removed. System ready.")

            else:
                no_detection_start_time = None

                draw_status(
                    frame,
                    "REMOVE SORTED OBJECT",
                    category=best_detection["category"],
                    confidence=best_detection["confidence"],
                    color=(0, 0, 255)
                )

        # ----------------------------------------------------
        # NORMAL DETECTION / 3-SECOND CONFIRMATION
        # ----------------------------------------------------

        elif best_detection is not None and not sorting:
            detected_category = best_detection["category"]
            detected_box = best_detection["box"]
            detected_confidence = best_detection["confidence"]

            same_category = (
                detected_category == previous_category
            )

            same_object = False

            if previous_box is not None:
                iou = calculate_iou(
                    previous_box,
                    detected_box
                )

                same_object = iou >= IOU_THRESHOLD

            # Start or continue confirmation only for same category
            # and same physical object location.
            if same_category and same_object:
                if confirmation_start_time is None:
                    confirmation_start_time = current_time

            else:
                previous_category = detected_category
                previous_box = detected_box
                confirmation_start_time = current_time

            previous_box = detected_box

            stable_seconds = (
                current_time - confirmation_start_time
            )

            draw_status(
                frame,
                "CONFIRMING OBJECT",
                category=detected_category,
                confidence=detected_confidence,
                stable_seconds=stable_seconds,
                color=(0, 255, 255)
            )

            # ------------------------------------------------
            # SORT AFTER 3 CONTINUOUS SECONDS
            # ------------------------------------------------

            if stable_seconds >= CONFIRMATION_SECONDS:
                sorting = True

                print("\n" + "=" * 55)
                print(
                    f"CONFIRMED FOR {CONFIRMATION_SECONDS:.1f} SECONDS"
                )
                print(
                    f"YOLO class: {best_detection['class_name']}"
                )
                print(
                    f"Final category: {detected_category}"
                )
                print(
                    f"Confidence: {detected_confidence:.2%}"
                )
                print("=" * 55 + "\n")

                draw_status(
                    frame,
                    "SORTING NOW",
                    category=detected_category,
                    confidence=detected_confidence,
                    stable_seconds=stable_seconds,
                    color=(0, 255, 0)
                )

                cv2.imshow(
                    "Waste Sorting System - CUDA",
                    frame
                )

                cv2.waitKey(1)

                send_command(detected_category)

                time.sleep(SORT_DELAY)

                send_command("REST")

                previous_category = None
                previous_box = None
                confirmation_start_time = None

                waiting_for_removal = True
                no_detection_start_time = None

                sorting = False

        # ----------------------------------------------------
        # NO DETECTION
        # ----------------------------------------------------

        else:
            previous_category = None
            previous_box = None
            confirmation_start_time = None

            draw_status(
                frame,
                "NO YOLO DETECTION",
                color=(0, 0, 255)
            )

        # ----------------------------------------------------
        # DISPLAY CAMERA WINDOW
        # ----------------------------------------------------

        cv2.imshow(
            "Waste Sorting System - CUDA",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("Q pressed. Exiting program.")
            break

finally:
    print("Cleaning up...")

    try:
        send_command("REST")
    except Exception:
        pass

    if cap.isOpened():
        cap.release()

    if esp32.is_open:
        esp32.close()

    cv2.destroyAllWindows()

    print("System closed safely.")





# import cv2
# import serial
# import time
# from ultralytics import YOLO


# # ============================================================
# # SETTINGS
# # ============================================================

# MODEL_PATH = "models/best.pt"

# SERIAL_PORT = "COM3"
# BAUD_RATE = 115200

# CAMERA_INDEX = 0

# # Lowered for debugging
# CONFIDENCE = 0.25

# REQUIRED_FRAMES = 5
# SORT_DELAY = 2.0


# # ============================================================
# # LOAD MODEL
# # ============================================================

# model = YOLO(MODEL_PATH)

# print("\nModel loaded successfully")
# print("Model classes:")
# print(model.names)
# print()


# # ============================================================
# # SERIAL CONNECTION
# # ============================================================

# esp32 = serial.Serial(
#     SERIAL_PORT,
#     BAUD_RATE,
#     timeout=1
# )

# time.sleep(2)

# print("ESP32 connected")


# # ============================================================
# # CAMERA
# # ============================================================

# cap = cv2.VideoCapture(CAMERA_INDEX)

# if not cap.isOpened():
#     print("Could not open camera")
#     exit()


# # ============================================================
# # VARIABLES
# # ============================================================

# last_detection = None
# frame_count = 0
# sorting = False


# # ============================================================
# # SEND COMMAND
# # ============================================================

# def send_command(command):

#     print("Sending:", command)

#     esp32.write((command + "\n").encode())

#     time.sleep(0.1)


# # ============================================================
# # CLASSIFY WASTE
# # ============================================================

# def classify_waste(class_name):

#     class_name = class_name.lower().strip()

#     if class_name == "plastic":
#         return "PLASTIC"

#     elif class_name in ["paper", "cardboard"]:
#         return "PAPER"

#     else:
#         return "OTHERS"


# # ============================================================
# # MAIN LOOP
# # ============================================================

# while True:

#     ret, frame = cap.read()

#     if not ret:
#         print("Camera frame error")
#         break


#     # ========================================================
#     # YOLO DETECTION
#     # ========================================================

#     results = model.predict(
#         source=frame,
#         conf=CONFIDENCE,
#         imgsz=640,
#         verbose=False
#     )


#     detected_waste = None
#     detected_confidence = 0


#     # ========================================================
#     # PROCESS ALL DETECTIONS
#     # ========================================================

#     for result in results:

#         for box in result.boxes:

#             confidence = float(box.conf[0])

#             class_id = int(box.cls[0])

#             class_name = model.names[class_id]

#             waste_type = classify_waste(class_name)


#             # ------------------------------------------------
#             # BOUNDING BOX
#             # ------------------------------------------------

#             x1, y1, x2, y2 = map(
#                 int,
#                 box.xyxy[0]
#             )


#             # ------------------------------------------------
#             # DRAW EVERY DETECTION
#             # ------------------------------------------------

#             cv2.rectangle(
#                 frame,
#                 (x1, y1),
#                 (x2, y2),
#                 (0, 255, 0),
#                 3
#             )


#             # ------------------------------------------------
#             # LABEL
#             # ------------------------------------------------

#             label = f"{class_name} {confidence:.2f}"

#             cv2.putText(
#                 frame,
#                 label,
#                 (x1, max(y1 - 10, 20)),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.7,
#                 (0, 255, 0),
#                 2
#             )


#             # ------------------------------------------------
#             # SHOW MAPPED CATEGORY
#             # ------------------------------------------------

#             category_label = f"-> {waste_type}"

#             cv2.putText(
#                 frame,
#                 category_label,
#                 (x1, y2 + 25),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.65,
#                 (255, 255, 0),
#                 2
#             )


#             # ------------------------------------------------
#             # PRINT DETECTION
#             # ------------------------------------------------

#             print(
#                 f"Detected: {class_name} | "
#                 f"Confidence: {confidence:.2f} | "
#                 f"Category: {waste_type}"
#             )


#             # ------------------------------------------------
#             # SELECT HIGHEST CONFIDENCE DETECTION
#             # ------------------------------------------------

#             if confidence > detected_confidence:

#                 detected_confidence = confidence
#                 detected_waste = waste_type


#     # ========================================================
#     # DETECTION STATUS
#     # ========================================================

#     if detected_waste is not None:

#         if detected_waste == last_detection:

#             frame_count += 1

#         else:

#             last_detection = detected_waste
#             frame_count = 1


#         cv2.putText(
#             frame,
#             f"Category: {detected_waste}",
#             (20, 40),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.9,
#             (0, 255, 0),
#             2
#         )


#         cv2.putText(
#             frame,
#             f"Confidence: {detected_confidence:.2f}",
#             (20, 75),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.7,
#             (0, 255, 255),
#             2
#         )


#         cv2.putText(
#             frame,
#             f"Stable: {frame_count}/{REQUIRED_FRAMES}",
#             (20, 110),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.7,
#             (0, 255, 255),
#             2
#         )


#         # ====================================================
#         # SORT
#         # ====================================================

#         if frame_count >= REQUIRED_FRAMES and not sorting:

#             sorting = True

#             send_command(detected_waste)

#             print(
#                 f"\n*** SORTING: {detected_waste} ***\n"
#             )

#             time.sleep(SORT_DELAY)

#             send_command("REST")

#             time.sleep(0.5)

#             last_detection = None
#             frame_count = 0

#             sorting = False


#     else:

#         last_detection = None
#         frame_count = 0

#         cv2.putText(
#             frame,
#             "No YOLO detection",
#             (20, 40),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.8,
#             (0, 0, 255),
#             2
#         )


#     # ========================================================
#     # SHOW CAMERA
#     # ========================================================

#     cv2.imshow(
#         "Waste Sorting System",
#         frame
#     )


#     # Q = quit

#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break


# # ============================================================
# # CLEANUP
# # ============================================================

# try:
#     send_command("REST")
# except:
#     pass

# cap.release()
# esp32.close()
# cv2.destroyAllWindows()