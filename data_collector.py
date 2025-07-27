import cv2
import mediapipe as mp
import numpy as np
import os
import csv
import time # Import time module for delays

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

    # Display a "Get Ready" message
    for i in range(3, 0, -1):
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame during countdown.")
            break
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, f"GET READY FOR {gesture_name} IN {i}...", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('Data Collector', frame)
        if cv2.waitKey(1000) & 0xFF == ord('q'): # Wait for 1 second
            print("Quitting data collection during countdown.")
            cap.release()
            cv2.destroyAllWindows()
            exit()
    
    # Small delay after countdown for user to finalize pose
    cv2.putText(frame, f"START {gesture_name} NOW!", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('Data Collector', frame)
    cv2.waitKey(500) # Show "START" message for half a second


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
                print("Failed to grab frame during data collection.")
                break

            frame = cv2.flip(frame, 1)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # Corrected line from previous error

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

            key = cv2.waitKey(1) & 0xFF # Process events continuously
            if key == ord('q'):
                print("Quitting data collection.")
                cap.release()
                cv2.destroyAllWindows()
                exit() # Exit the whole script if 'q' is pressed

            # Add a small delay to avoid consuming 100% CPU and to allow the OS to catch up
            # This is especially helpful if not enough frames are being processed per second
            # and 'count' increases too fast
            # time.sleep(0.001) # Experiment with this value, or remove if not needed

        print(f"Finished collecting data for {gesture_name}.")

cap.release()
cv2.destroyAllWindows()
print("Data collection complete for all gestures.")