import cv2
import numpy as np
import mediapipe as mp
import face_recognition
from pathlib import Path
from collections import Counter

EMBEDDINGS_DIR = Path(__file__).resolve().parent / "embeddings"

# Load known embeddings once
Know_embed = []
Know_name = []

for file in EMBEDDINGS_DIR.glob("*.npy"):
    if file.suffix == ".npy":
        name = file.stem
        data = np.load(file)

        if data.ndim == 1:
            Know_embed.append(data)
            Know_name.append(name)
        else:
            for emb in data:
                Know_embed.append(emb)
                Know_name.append(name)

Threshold = 0.4


def detect_face():
    """
    Opens camera, detects face, returns recognized name
    Press 'q' to exit
    """

    mp_face_detection = mp.solutions.face_detection

    name_hist = []
    hist_size = 7
    frame_count = 0
    last_name = "Unknown"

    def update_identity(current_name):
        nonlocal name_hist, last_name

        if current_name is None:
            return last_name

        name_hist.append(current_name)

        if len(name_hist) > hist_size:
            name_hist.pop(0)

        stable_name = Counter(name_hist).most_common(1)[0][0]
        last_name = stable_name

        return last_name

    with mp_face_detection.FaceDetection(
        model_selection=1,
        min_detection_confidence=0.5
    ) as face_detection:

        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Camera not opened")
            return None

        print("Camera started... Press 'q' to stop")

        while True:
            ret, frame = cap.read()
            frame_count += 1

            if not ret:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(rgb)

            if results.detections:
                det = results.detections[0]
                bbox = det.location_data.relative_bounding_box
                h, w, _ = frame.shape

                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                box_w = int(bbox.width * w)
                box_h = int(bbox.height * h)

                top = max(0, y)
                left = max(0, x)
                right = min(w, x + box_w)
                bottom = min(h, y + box_h)

                current_name = None

                if frame_count % 5 == 0:
                    encodings = face_recognition.face_encodings(
                        rgb,
                        known_face_locations=[(top, right, bottom, left)]
                    )

                    if encodings:
                        current_embedding = encodings[0]

                        distance = face_recognition.face_distance(
                            Know_embed,
                            current_embedding
                        )

                        best_index = np.argmin(distance)
                        best_distance = distance[best_index]

                        if best_distance < Threshold:
                            current_name = Know_name[best_index]
                        else:
                            current_name = "Unknown"
                    else:
                        current_name = "Unknown"

                stable_name = update_identity(current_name)

                # Draw (optional UI)
                cv2.rectangle(frame, (x, y), (x + box_w, y + box_h), (0, 255, 0), 2)
                cv2.putText(frame, stable_name, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

                cv2.imshow("Face Recognition", frame)

                #  RETURN when stable name found
                if stable_name != "Unknown":
                    cap.release()
                    cv2.destroyAllWindows()
                    return stable_name

            cv2.imshow("Face Recognition", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        return None
