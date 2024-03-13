
from flask import Flask, render_template, request
from flask import Flask, render_template
import pyttsx3
import speech_recognition as sr
from threading import Thread
import cv2
import mediapipe as mp
import numpy as np
import dlib


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/blind_only')
def blind_only():
    return render_template('main_page_steps.html')

@app.route('/dumb')
def blind_only():
    return render_template('main_page_steps.html')

def hand_gesture_reg():
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

def eye_gest_reg():
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    def process_eye_gestures(image_data):
        img = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        for face in faces:
            landmarks = predictor(gray, face)
            left_eye_landmarks = []
            right_eye_landmarks = []

        # Extract left and right eye landmarks
            for n in range(36, 42):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                left_eye_landmarks.append((x, y))

            for n in range(42, 48):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                right_eye_landmarks.append((x, y))

        # Calculate eye aspect ratio (EAR) for left eye
            left_eye_width = cv2.norm(left_eye_landmarks[3], left_eye_landmarks[0])
            left_eye_height1 = cv2.norm(left_eye_landmarks[1], left_eye_landmarks[5])
            left_eye_height2 = cv2.norm(left_eye_landmarks[2], left_eye_landmarks[4])
            left_eye_ear = (left_eye_height1 + left_eye_height2) / (2 * left_eye_width)

        # Calculate eye aspect ratio (EAR) for right eye
            right_eye_width = cv2.norm(right_eye_landmarks[3], right_eye_landmarks[0])
            right_eye_height1 = cv2.norm(right_eye_landmarks[1], right_eye_landmarks[5])
            right_eye_height2 = cv2.norm(right_eye_landmarks[2], right_eye_landmarks[4])
            right_eye_ear = (right_eye_height1 + right_eye_height2) / (2 * right_eye_width)

        # Example: Determine if eyes are open or closed based on EAR threshold
            if left_eye_ear < 0.2 and right_eye_ear < 0.2:
                print("Eyes closed")
            else:
                print("Eyes open")

@app.route('/finger_counter')
def finger_count():
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

        if finger_count == 1:
            order="COFFEE"
        else:
            order="TEA"
        return render_template('main_page_steps.html',order = order)


    
if __name__ == '__main__':
    app.run(debug=True)