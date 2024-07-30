import cv2
import mediapipe as mp
import pyautogui
import time

# Function to count the number of extended fingers
def count_fingers(hand_landmarks):
    finger_count = 0
    threshold = (hand_landmarks.landmark[0].y * 100 - hand_landmarks.landmark[9].y * 100) / 2

    if (hand_landmarks.landmark[5].y * 100 - hand_landmarks.landmark[8].y * 100) > threshold:
        finger_count += 1
    if (hand_landmarks.landmark[9].y * 100 - hand_landmarks.landmark[12].y * 100) > threshold:
        finger_count += 1
    if (hand_landmarks.landmark[13].y * 100 - hand_landmarks.landmark[16].y * 100) > threshold:
        finger_count += 1
    if (hand_landmarks.landmark[17].y * 100 - hand_landmarks.landmark[20].y * 100) > threshold:
        finger_count += 1
    if (hand_landmarks.landmark[5].x * 100 - hand_landmarks.landmark[4].x * 100) > 6:
        finger_count += 1

    return finger_count

# Initialize the video capture and MediaPipe Hands
cap = cv2.VideoCapture(0)
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)

# Variables for gesture and swipe detection
start_init = False
previous_finger_count = -1
previous_x = None
previous_time = None

print("Starting hand gesture recognition. Press 'Esc' to exit.")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame. Exiting...")
        break
    
    # Flip the frame horizontally for a mirror-like effect
    frame = cv2.flip(frame, 1)
    
    # Process the frame with MediaPipe Hands
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    # If hand landmarks are detected
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]

        # Count the number of extended fingers
        finger_count = count_fingers(hand_landmarks)

        # Check if the finger count has changed
        if previous_finger_count != finger_count:
            if not start_init:
                start_time = time.time()
                start_init = True
            elif time.time() - start_time > 0.2:
                # Perform actions based on finger count
                if finger_count == 1:
                    pyautogui.press("right")
                elif finger_count == 2:
                    pyautogui.press("left")
                elif finger_count == 3:
                    pyautogui.press("up")
                elif finger_count == 4:
                    pyautogui.press("down")
                elif finger_count == 5:
                    pyautogui.press("space")
                
                previous_finger_count = finger_count
                start_init = False

        # Detect swipes based on the movement of the wrist landmark
        current_x = hand_landmarks.landmark[0].x * 100
        current_time = time.time()

        if previous_x is not None and previous_time is not None:
            delta_x = current_x - previous_x
            delta_time = current_time - previous_time

            if abs(delta_x) > 20 and delta_time < 0.5:
                if delta_x > 0:
                    pyautogui.hotkey('shift', 'n')  # Swipe right
                elif delta_x < 0:
                    pyautogui.hotkey('shift', 'p')  # Swipe left
                previous_x = None
                previous_time = None
            else:
                previous_x = current_x
                previous_time = current_time
        else:
            previous_x = current_x
            previous_time = current_time

        # Draw hand landmarks on the frame
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the resulting frame
    cv2.imshow("Hand Gesture Recognition", frame)

    # Break the loop if 'Esc' is pressed or the window is closed
    if cv2.getWindowProperty('Hand Gesture Recognition', cv2.WND_PROP_VISIBLE) < 1 or cv2.waitKey(1) == 27:
        print("Exiting hand gesture recognition.")
        break

# Release the capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
