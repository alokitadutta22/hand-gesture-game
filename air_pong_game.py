import cv2
import mediapipe as mp
import pygame
import sys # For clean exit
import math # For distance calculation for gestures

# --- MediaPipe Setup ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# --- Pygame Setup ---
pygame.init()

SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Air Pong (Hand Controlled)")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# Game elements (initial values, will be reset by reset_game())
paddle = pygame.Rect(0, 0, 0, 0) # Placeholder, actual values set in reset_game
ball = pygame.Rect(0, 0, 0, 0)   # Placeholder
ball_dx, ball_dy = 0, 0          # Placeholder
score = 0
game_over_reason = "" # To display why game ended

# Game state management
START_SCREEN = 0
PLAYING = 1
GAME_OVER = 2
game_state = START_SCREEN # Start with the start screen

# Paddle properties (constants)
PADDLE_WIDTH, PADDLE_HEIGHT = 150, 20
PADDLE_SPEED_DAMPING = 0.2 # Damping factor for smooth movement

# Ball properties (constants)
BALL_SIZE = 20
INITIAL_BALL_SPEED = 7  # Increased from 5

# Font setup
font = pygame.font.Font(None, 74)
small_font = pygame.font.Font(None, 50)

# --- Gesture Detection Variables ---
is_fist_closed = False # State to detect if fist is currently closed
FIST_THRESHOLD = 0.07 # Normalized distance threshold to consider a fist closed (tune this!)
# This threshold is the average distance from the tip of each finger to its MCP joint
# A smaller value means a tighter fist is required.

# --- Function to calculate distance between two landmarks ---
def calculate_distance(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

# --- Function to detect if a hand is a fist ---
def is_hand_fist(hand_landmarks):
    # Get landmarks for thumb, index, middle, ring, pinky fingers
    # MCP (Metacarpophalangeal Joint) - the knuckles at the base of the fingers
    # PIP (Proximal Interphalangeal Joint) - the middle knuckles
    # TIP (Tip of the finger)

    # For a fist, finger tips should be close to the palm/MCP joints
    # Let's check distance from tips to their respective MCPs
    fingers_folded = []
    
    # Check Index Finger: Tip (8) close to PIP (6) and MCP (5)
    # Check Middle Finger: Tip (12) close to PIP (10) and MCP (9)
    # Check Ring Finger: Tip (16) close to PIP (14) and MCP (13)
    # Check Pinky Finger: Tip (20) close to PIP (18) and MCP (17)
    
    # We'll use a simpler check: distance from tip to MCP of each finger.
    # If the distance is small, the finger is likely folded.
    
    landmark_indices = [
        (mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_MCP), # 8-5
        (mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_MCP), # 12-9
        (mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_MCP), # 16-13
        (mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_MCP) # 20-17
    ]
    
    all_fingers_folded = True
    for tip_idx, mcp_idx in landmark_indices:
        tip = hand_landmarks.landmark[tip_idx]
        mcp = hand_landmarks.landmark[mcp_idx]
        
        dist = calculate_distance(tip, mcp)
        
        # You'll need to tune this threshold based on your hand and webcam setup
        # Smaller value means fingers must be very close to palm
        if dist > FIST_THRESHOLD:
            all_fingers_folded = False
            break # At least one finger is extended, so not a fist

    # Also check thumb position relative to palm or other fingers for a definite fist
    # For a "true" fist, the thumb tip (4) should be close to the index finger MCP (5) or middle finger MCP (9)
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    
    thumb_dist_to_palm = calculate_distance(thumb_tip, index_mcp)
    
    # This threshold also needs tuning. If the thumb is too far, it's not a fist.
    if thumb_dist_to_palm > FIST_THRESHOLD * 2: # Thumb can be a bit further than fingers
        all_fingers_folded = False

    return all_fingers_folded

# --- Function to reset the game state ---
def reset_game():
    global paddle, ball, ball_dx, ball_dy, score, game_over, game_over_reason, game_state

    # Paddle reset
    paddle.update(SCREEN_WIDTH // 2 - PADDLE_WIDTH // 2, SCREEN_HEIGHT - 50, PADDLE_WIDTH, PADDLE_HEIGHT)

    # Ball reset
    ball.update(SCREEN_WIDTH // 2 - BALL_SIZE // 2, SCREEN_HEIGHT // 2 - BALL_SIZE // 2, BALL_SIZE, BALL_SIZE)
    ball_dx = INITIAL_BALL_SPEED # Reset ball speed and direction
    ball_dy = INITIAL_BALL_SPEED

    score = 0
    game_over = False
    game_over_reason = ""
    game_state = PLAYING # Set game state to playing after reset

# --- Webcam Setup ---
cap = cv2.VideoCapture(0) # 0 for default webcam

if not cap.isOpened():
    print("Error: Could not open webcam.")
    print("Please ensure your webcam is connected and not in use by another application.")
    pygame.quit()
    sys.exit()

print("Webcam opened successfully. Air Pong game started.")
print("Control paddle with your hand (Index finger).")
print("From Game Over screen, make a FIST gesture to play again.")
print("Close Pygame window or press 'q' in webcam window to quit.")

# --- Game Loop ---
running = True
clock = pygame.time.Clock() # To control frame rate

# Initial game reset to set up elements for start screen
reset_game()
game_state = START_SCREEN # Override to start screen initially

while running:
    # 1. Event Handling (Pygame)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # 2. Webcam Capture and MediaPipe Processing
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame. Exiting.")
        running = False
        break

    frame = cv2.flip(frame, 1) # Flip for selfie-view
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # Reset gesture state each frame before re-evaluation
    current_fist_closed = False

    # 3. Process Hand Tracking Data and Update Game State (Hand Control & Gestures)
    if results.multi_hand_landmarks:
        # Assuming we only care about the first detected hand
        hand_landmarks = results.multi_hand_landmarks[0]

        # Always draw hand landmarks for debugging the webcam feed
        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )
        
        # --- Check for Fist Gesture ---
        if is_hand_fist(hand_landmarks):
            current_fist_closed = True
            if not is_fist_closed and game_state == GAME_OVER:
                # Fist just closed and game is over -> Restart game
                reset_game()
                print("Game restarted by FIST gesture!")
            is_fist_closed = True # Update state for next frame
        else:
            is_fist_closed = False # Fist is open

        # --- Paddle Control (Only when PLAYING) ---
        if game_state == PLAYING:
            # Get the X-coordinate of the index finger tip (landmark 8)
            index_finger_x_norm = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x
            
            # Map normalized X to Pygame screen width
            target_paddle_center_x = int(index_finger_x_norm * SCREEN_WIDTH)

            # Smoothly move the paddle towards the target X position
            paddle.centerx += (target_paddle_center_x - paddle.centerx) * PADDLE_SPEED_DAMPING

            # Keep paddle within screen bounds
            if paddle.left < 0:
                paddle.left = 0
            if paddle.right > SCREEN_WIDTH:
                paddle.right = SCREEN_WIDTH
    else:
        # No hand detected: paddle stops, or you could add a "pause" behavior
        if game_state == PLAYING:
            # Optionally, you can stop the ball or pause the game if no hand is detected
            pass


    # 4. Game Logic Update (Ball Movement, Collisions, Score)
    if game_state == PLAYING:
        ball.x += ball_dx
        ball.y += ball_dy

        # Ball collision with walls
        if ball.left < 0 or ball.right > SCREEN_WIDTH:
            ball_dx *= -1 # Reverse X direction
        if ball.top < 0:
            ball_dy *= -1 # Reverse Y direction

        # Ball collision with paddle
        if ball.colliderect(paddle):
            # To prevent ball getting stuck in paddle, adjust its position slightly
            ball.bottom = paddle.top
            ball_dy *= -1.1  # Increase speed by 10% each hit
            ball_dx *= 1.1   # Increase horizontal speed too
            score += 1 # Increase score

        # Ball goes past bottom (Game Over)
        if ball.top > SCREEN_HEIGHT:
            game_over = True
            game_over_reason = "Ball missed!"
            game_state = GAME_OVER

    # 5. Drawing and Display (Pygame)
    screen.fill(BLACK) # Clear screen

    if game_state == START_SCREEN:
        title_text = font.render("AIR PONG", True, WHITE)
        instruction_text = small_font.render("Show your hand to start!", True, GREEN)
        screen.blit(title_text, (SCREEN_WIDTH // 2 - title_text.get_width() // 2, SCREEN_HEIGHT // 2 - 50))
        screen.blit(instruction_text, (SCREEN_WIDTH // 2 - instruction_text.get_width() // 2, SCREEN_HEIGHT // 2 + 10))

        # Check for hand presence to start the game
        if results.multi_hand_landmarks:
            game_state = PLAYING

    elif game_state == PLAYING:
        pygame.draw.rect(screen, WHITE, paddle) # Draw paddle
        pygame.draw.ellipse(screen, WHITE, ball) # Draw ball (ellipse makes it look round)

        # Display score
        score_text = font.render(f"Score: {score}", True, WHITE)
        screen.blit(score_text, (10, 10))

    elif game_state == GAME_OVER:
        game_over_text = font.render("GAME OVER!", True, RED)
        reason_text = small_font.render(game_over_reason, True, WHITE)
        score_display_text = small_font.render(f"Final Score: {score}", True, GREEN)
        play_again_text = small_font.render("Make a FIST to Play Again!", True, BLUE)
        exit_text = small_font.render("Close window or press 'q' to quit", True, WHITE)

        screen.blit(game_over_text, (SCREEN_WIDTH // 2 - game_over_text.get_width() // 2, SCREEN_HEIGHT // 2 - 120))
        screen.blit(reason_text, (SCREEN_WIDTH // 2 - reason_text.get_width() // 2, SCREEN_HEIGHT // 2 - 60))
        screen.blit(score_display_text, (SCREEN_WIDTH // 2 - score_display_text.get_width() // 2, SCREEN_HEIGHT // 2))
        screen.blit(play_again_text, (SCREEN_WIDTH // 2 - play_again_text.get_width() // 2, SCREEN_HEIGHT // 2 + 60))
        screen.blit(exit_text, (SCREEN_WIDTH // 2 - exit_text.get_width() // 2, SCREEN_HEIGHT // 2 + 120))


    pygame.display.flip() # Update the full display Surface to the screen

    # 6. Display Webcam Debug Feed (Optional but Recommended)
    cv2.imshow('Webcam Feed (Hand Tracking Debug)', frame)

    # Check for 'q' key press in OpenCV window to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        running = False


    clock.tick(60) # Control frame rate (60 FPS)

# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()
pygame.quit()
sys.exit()