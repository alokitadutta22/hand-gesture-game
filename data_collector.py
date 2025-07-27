import cv2
import mediapipe as mp
import numpy as np
import os
import csv

# --- Setup MediaPipe Hands ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_drawing = mp.solutions.drawing_utils

# --- Data Collection Setup ---
DATA_DIR = "gesture_data" # Folder to save data
gestures = ["FIST", "OPEN_PALM", "V_SIGN", "ILY_SIGN"] # Your chosen gestures
num_samples_per_gesture = 500 # Number of frames to capture for each gesture

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
for gesture in gestures:
    os.makedirs(os.path.join(DATA_DIR, gesture), exist_ok=True)

def extract_landmark_features(hand_landmarks):
    # Flatten all 21 (x, y, z) coordinates into a 63-element array
    features = []
    for landmark in hand_landmarks.landmark:
        features.extend([landmark.x, landmark.y, landmark.z])
    return np.array(features)

# --- Main Loop for Data Collection ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print(f"Collecting {num_samples_per_gesture} samples for each gesture.")
print("Hold the gesture steady when the counter starts for that gesture.")
print("Press 'q' to quit at any time.")

for gesture_name in gestures:
    print(f"\n--- Collecting data for: {gesture_name} ---")
    input(f"Press Enter to start collecting {gesture_name} data...") # Wait for user to get ready

    count = 0
    csv_file_path = os.path.join(DATA_DIR, f"{gesture_name}.csv")

    # Open CSV file to write data
    with open(csv_file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        # Write header (landmark_x_0, landmark_y_0, ..., landmark_z_20, label)
        header = [f"lm_{i}_{coord}" for i in range(21) for coord in ['x', 'y', 'z']]
        header.append("label")
        csv_writer.writerow(header)

        while count < num_samples_per_gesture:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            frame = cv2.flip(frame, 1)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convert BGR to RGB for MediaPipe
            results = hands.process(image)
            
            # Convert the RGB image back to BGR for OpenCV display.
            # THIS IS THE LINE THAT HAD THE TYPO:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # CORRECTED LINE

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    features = extract_landmark_features(hand_landmarks)
                    row = features.tolist() + [gesture_name]
                    csv_writer.writerow(row)
                    count += 1

            cv2.putText(image, f"Gesture: {gesture_name}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(image, f"Samples: {count}/{num_samples_per_gesture}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2, cv2.LINE_AA)

            cv2.imshow('Data Collector', image)

            if cv2.waitKey(5) & 0xFF == ord('q'):
                print("Quitting data collection.")
                cap.release()
                cv2.destroyAllWindows()
                exit() # Exit the whole script if 'q' is pressed

        print(f"Finished collecting data for {gesture_name}.")

cap.release()
cv2.destroyAllWindows()
print("Data collection complete for all gestures.")