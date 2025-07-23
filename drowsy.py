import cv2
import mediapipe as mp
import numpy as np
import pygame

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Constants
EAR_THRESHOLD = 0.22
MAR_THRESHOLD = 0.65
EYE_CLOSURE_THRESHOLD = 70
YAWN_THRESHOLD = 30

frame_counter_eye = 0
frame_counter_yawn = 0

# Landmark indices
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
MOUTH_IDX = [78, 308, 13, 14, 87, 317]

def aspect_ratio(points):
    A = np.linalg.norm(points[1] - points[5])
    B = np.linalg.norm(points[2] - points[4])
    C = np.linalg.norm(points[0] - points[3])
    return (A + B) / (2.0 * C)

# UI utility
def draw_status_box(frame, text, color, position=(20, 20), bg_color=(0, 0, 0)):
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    x, y = position
    cv2.rectangle(frame, (x - 10, y - 30), (x + w + 10, y + 10), bg_color, -1)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

def draw_progress_bar(frame, progress, position=(20, 100), width=300, height=10, color=(0, 255, 0)):
    # Cap the progress to ensure it doesn't overflow
    progress = min(progress, 1.0)
    cv2.rectangle(frame, position, (position[0] + width, position[1] + height), (0, 0, 0), -1)  # Background
    cv2.rectangle(frame, position, (position[0] + int(progress * width), position[1] + height), color, -1)

# Video capture
cap = cv2.VideoCapture(0)

# Initialize pygame mixer
pygame.init()
pygame.mixer.init()

# Load sound effects
alert_sound = pygame.mixer.Sound("alert.mp3")
face_not_detected_sound = pygame.mixer.Sound("face_not_detected.mp3")

# Create dedicated channels
alert_channel = pygame.mixer.Channel(0)
face_channel = pygame.mixer.Channel(1)


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Webcam not detected")
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    drowsy = False
    face_detected = False
    h, w, _ = frame.shape

    if results.multi_face_landmarks:
        face_detected = True
        for landmarks in results.multi_face_landmarks:
            def get_landmark_coords(idxs):
                return np.array([[int(landmarks.landmark[i].x * w), int(landmarks.landmark[i].y * h)] for i in idxs], dtype=np.float32)

            left_eye = get_landmark_coords(LEFT_EYE_IDX)
            right_eye = get_landmark_coords(RIGHT_EYE_IDX)
            mouth = get_landmark_coords(MOUTH_IDX)

            left_ear = aspect_ratio(left_eye)
            right_ear = aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0
            mar = aspect_ratio(mouth)

            # EAR and MAR conditions
            if ear < EAR_THRESHOLD:
                frame_counter_eye += 1
            else:
                frame_counter_eye = 0

            if mar > MAR_THRESHOLD:
                frame_counter_yawn += 1
            else:
                frame_counter_yawn = 0

            # Drowsy logic
            if frame_counter_eye >= EYE_CLOSURE_THRESHOLD or frame_counter_yawn >= YAWN_THRESHOLD:
                drowsy = True

            # Draw facial landmarks (optional)
            for pt in np.concatenate((left_eye, right_eye), axis=0):
                cv2.circle(frame, tuple(pt.astype(int)), 2, (0, 255, 0), -1)
            for pt in mouth:
                cv2.circle(frame, tuple(pt.astype(int)), 2, (255, 0, 0), -1)

            # Overlay EAR/MAR values
            draw_status_box(frame, f"EAR: {ear:.2f}", (255, 255, 255), (20, h - 60))
            draw_status_box(frame, f"MAR: {mar:.2f}", (255, 255, 255), (200, h - 60))

            # Progress Bar for Drowsiness Detection (Eye/ Yawn threshold tracking)
            progress = max(frame_counter_eye, frame_counter_yawn) / max(EYE_CLOSURE_THRESHOLD, YAWN_THRESHOLD)
            draw_progress_bar(frame, progress, (20, 120))

   # Alerts
    if not face_detected:
        draw_status_box(frame, "No Face Detected", (255, 0, 0), (20, 50), (0, 0, 0))
        if not face_channel.get_busy():
            face_channel.play(face_not_detected_sound)
    elif drowsy:
        draw_status_box(frame, "DROWSY ALERT!", (255, 255, 255), (20, 50), (0, 0, 255))
        if not alert_channel.get_busy():
            alert_channel.play(alert_sound)
        cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 5)
    else:
        draw_status_box(frame, "Status: Awake", (0, 255, 0), (20, 50), (0, 0, 0))

    # Drowsiness Meter Text
    draw_status_box(frame, "Drowsiness Meter", (255, 255, 255), (20, 90), (0, 0, 0))

    # Show frame
    cv2.imshow("Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()