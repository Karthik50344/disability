import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

def process_hand_gestures(image_data):
    img = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for idx, landmark in enumerate(hand_landmarks.landmark):
                # Extracting landmark coordinates
                cx, cy = int(landmark.x * img.shape[1]), int(landmark.y * img.shape[0])
                # Draw the landmarks on the image (optional)
                cv2.circle(img, (cx, cy), 5, (255, 0, 0), -1)
            # Process hand landmarks here...
            # Example: Calculate distance between specific landmarks
            index_finger_tip = (hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * img.shape[1],
                                hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * img.shape[0])
            thumb_tip = (hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * img.shape[1],
                         hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * img.shape[0])
            distance = np.sqrt((index_finger_tip[0] - thumb_tip[0])**2 + (index_finger_tip[1] - thumb_tip[1])**2)
            print("Distance between index finger tip and thumb tip:", distance)
