# Drowsiness-Detection-System
Drowsiness detection is a critical application aimed at enhancing safety in
environments where human alertnessis essential,such asin driving and operating machinery.
This project presents a real-time drowsiness detection system utilizing computer vision
techniques to monitor fatigue indicators through facial landmarks.

  The system focuses on analyzing behavioral cues such as eye closure duration and
yawning frequency by calculating Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR)
from video frames. Using MediaPipe's Face Mesh, the model detects facial features with
high accuracy, enabling precise measurement of these indicators. Audio alerts are generated
through a Pygame-based system to notify users when drowsiness is detected.

  Unlike traditional machine learning approaches, this implementation relies on
threshold-based logic to classify drowsiness levels, reducing computational overhead and
improving responsiveness. Real-time video capture and OpenCV-based processing allow the
system to operate efficiently even under varied lighting conditions. The integration of visual
feedback, such as status indicators and progress bars, enhances user interaction and
situational awareness.
  
  Although the current system does not incorporate physiological signals or deep
learning, it demonstrates high reliability in detecting early signs of fatigue. The solution is
particularly well-suited for embedded systems or low-resource environments.
