import cv2
import mediapipe as mp
import numpy as np # Import numpy

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_drawing = mp.solutions.drawing_utils

def get_hand_gesture(hand_landmarks):
    landmarks = hand_landmarks.landmark

    # Helper function to calculate distance between two landmarks
    def get_distance(p1, p2):
        return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

    # --- EXTRACT LANDMARK COORDINATES FOR FINGERS ---
    # Thumb
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = landmarks[mp_hands.HandLandmark.THUMB_IP]
    thumb_mcp = landmarks[mp_hands.HandLandmark.THUMB_MCP]

    # Index Finger
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_pip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP]
    index_mcp = landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP]

    # Middle Finger
    middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_pip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
    middle_mcp = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]

    # Ring Finger
    ring_tip = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP]
    ring_pip = landmarks[mp_hands.HandLandmark.RING_FINGER_PIP]
    ring_mcp = landmarks[mp_hands.HandLandmark.RING_FINGER_MCP]

    # Pinky Finger
    pinky_tip = landmarks[mp_hands.HandLandmark.PINKY_TIP]
    pinky_pip = landmarks[mp_hands.HandLandmark.PINKY_PIP]
    pinky_mcp = landmarks[mp_hands.HandLandmark.PINKY_MCP]

    # Use .y and .x properties for direct access
    index_tip_y = index_tip.y
    index_pip_y = index_pip.y
    index_mcp_y = index_mcp.y

    middle_tip_y = middle_tip.y
    middle_pip_y = middle_pip.y
    middle_mcp_y = middle_mcp.y

    ring_tip_y = ring_tip.y
    ring_pip_y = ring_pip.y
    ring_mcp_y = ring_mcp.y

    pinky_tip_y = pinky_tip.y
    pinky_pip_y = pinky_pip.y
    pinky_mcp_y = pinky_mcp.y

    thumb_tip_y = thumb_tip.y
    thumb_ip_y = thumb_ip.y
    thumb_mcp_y = thumb_mcp.y

    thumb_tip_x = thumb_tip.x
    thumb_mcp_x = thumb_mcp.x
    # --- END EXTRACT LANDMARK COORDINATES ---


    # Thresholds (you'll need to experiment with these values based on your hand and camera)
    # A higher Y value means lower on the screen.
    finger_curl_threshold = 0.04 # How much lower the tip needs to be than PIP/MCP to be considered curled

    index_curled = index_tip_y > index_pip_y + finger_curl_threshold
    middle_curled = middle_tip_y > middle_pip_y + finger_curl_threshold
    ring_curled = ring_tip_y > ring_pip_y + finger_curl_threshold
    pinky_curled = pinky_tip_y > pinky_pip_y + finger_curl_threshold

    # Heuristic for thumb curled (tucked in) - adjust as needed
    # Simple check: if thumb tip is close to the palm or lower than its MCP (depends on hand orientation)
    # A more robust check might involve angles.
    thumb_curled = (
        get_distance(landmarks[mp_hands.HandLandmark.THUMB_TIP], landmarks[mp_hands.HandLandmark.WRIST]) <
        get_distance(landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP], landmarks[mp_hands.HandLandmark.WRIST]) * 0.7 # Example
    )


    # Fist detection
    if all([index_curled, middle_curled, ring_curled, pinky_curled]) and thumb_curled:
        return "FIST (PUNCH)"

    # Open Palm detection
    # Ensure all fingers are generally extended and the thumb is also extended (not curled)
    # Use a threshold for "extended" (tip is significantly higher/further from palm than PIP)
    extended_finger_threshold = -0.02 # A small negative value means tip is above PIP

    index_extended = index_tip_y < index_pip_y + extended_finger_threshold
    middle_extended = middle_tip_y < middle_pip_y + extended_finger_threshold
    ring_extended = ring_tip_y < ring_pip_y + extended_finger_threshold
    pinky_extended = pinky_tip_y < pinky_pip_y + extended_finger_threshold

    thumb_extended_x = abs(thumb_tip_x - thumb_mcp_x) > 0.05 # Thumb extends outwards on X
    thumb_extended_y = thumb_tip_y < thumb_ip_y # Thumb tip is higher than IP
    thumb_extended = thumb_extended_x and thumb_extended_y # Both X and Y conditions

    if all([index_extended, middle_extended, ring_extended, pinky_extended]) and thumb_extended:
        return "OPEN_PALM (BLOCK)"

    # Example: 'V' sign (Index and Middle extended, others curled, thumb somewhat out)
    if index_extended and middle_extended and \
       ring_curled and pinky_curled and \
       not thumb_curled: # Thumb should not be explicitly curled, might be slightly extended or neutral
        return "V_SIGN (KICK)"

    # Example: 'ILY' sign (Index and Pinky extended, Thumb extended, Middle and Ring curled)
    # This one is tricky as thumb also extends
    # ILY: Index, Pinky, and Thumb extended. Middle & Ring curled.
    if index_extended and pinky_extended and thumb_extended and \
       middle_curled and ring_curled:
       return "ILY_SIGN (SPECIAL)"


    return "UNKNOWN"


cap = cv2.VideoCapture(0) # 0 for default webcam

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    frame = cv2.flip(frame, 1)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image and find hands.
    results = hands.process(image)

    # Convert the RGB image back to BGR for OpenCV display.
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    current_gesture = "NO HAND DETECTED"

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            current_gesture = get_hand_gesture(hand_landmarks)

    # Display the recognized gesture on the frame
    cv2.putText(image, f"Gesture: {current_gesture}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('Hand Gesture Recognizer', image)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()