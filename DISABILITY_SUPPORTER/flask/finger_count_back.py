import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

def count_fingers(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    finger_count = 0
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Count fingers using hand landmarks
            # Assuming the hand is open with the palm facing the camera
            # We can count fingers based on the landmark positions
            # For simplicity, we'll count fingers based on the position of the index finger tip
            
            # Get the index finger tip landmark coordinates
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            
            # Get the middle finger tip landmark coordinates
            middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            
            # Get the ring finger tip landmark coordinates
            ring_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
            
            # Get the little finger tip landmark coordinates
            little_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
            
            # Get the thumb tip landmark coordinates
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            
            # Check if fingers are extended (landmark y-coordinates of fingers are above the landmark of the palm)
            if index_finger_tip.y < hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y:
                finger_count += 1
            if middle_finger_tip.y < hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y:
                finger_count += 1
            if ring_finger_tip.y < hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y:
                finger_count += 1
            if little_finger_tip.y < hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y:
                finger_count += 1
            if thumb_tip.y < hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y:
                finger_count += 1

    return finger_count
