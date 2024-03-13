import numpy as np
from flask import Flask, render_template, request
from textblob import TextBlob


def eye_aspect_ratio(eye):
    # Compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dlib.euclidean_distance(eye[1], eye[5])
    B = dlib.euclidean_distance(eye[2], eye[4])

    # Compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dlib.euclidean_distance(eye[0], eye[3])

    # Compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    return ear

# Define the constants for eye blink detection
EYE_AR_THRESHOLD = 0.3
EYE_AR_CONSEC_FRAMES = 3

# Initialize the frame counters and the total number of blinks
COUNTER = 0
TOTAL = 0

# Initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Start the video stream
video_capture = cv2.VideoCapture(0)

while True:
    # Capture the frame from the video stream
    ret, frame = video_capture.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = detector(gray, 0)

    for face in faces:
        # Determine the facial landmarks for the face region
        shape = predictor(gray, face)
        shape = dlib.full_object_detection(frame, shape)

        # Extract the left and right eye coordinates
        left_eye = [shape.part(i) for i in range(36, 42)]
        right_eye = [shape.part(i) for i in range(42, 48)]

        # Calculate the eye aspect ratio for both eyes
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)

        # Average the eye aspect ratio for both eyes
        ear = (left_ear + right_ear) / 2.0

        # Draw the eyes' landmarks on the frame
        for eye in [left_eye, right_eye]:
            for i in range(0, 6):
                cv2.line(frame, (eye[i].x, eye[i].y), (eye[i + 1].x, eye[i + 1].y), (0, 255, 0), 1)

        # Check if the eye aspect ratio is below the threshold
        if ear < EYE_AR_THRESHOLD:
            COUNTER += 1
        else:
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                TOTAL += 1
            COUNTER = 0

        # Display the eye aspect ratio and the total number of blinks on the frame
        cv2.putText(frame, f"Eye Aspect Ratio: {ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Total Blinks: {TOTAL}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow("Eye Blink Detection", frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video stream and close all windows
video_capture.release()
cv2.destroyAllWindows()


import cv2
import numpy as np

def find_fingers_count(hand_contour):
    # Convex hull to get the outer points of the hand contour
    hull = cv2.convexHull(hand_contour, returnPoints=False)

    # Defects will store the distances between the farthest point and the convex hull
    defects = []

    # Calculate the defects (the depth of each finger)
    if len(hull) >= 3:
        defects = cv2.convexityDefects(hand_contour, hull)

    finger_count = 0

    if defects is not None:
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(hand_contour[s][0])
            end = tuple(hand_contour[e][0])
            far = tuple(hand_contour[f][0])

            # Filter out the defects that are not fingers
            if d > 10000:
                finger_count += 1

    return finger_count + 1  # Add 1 for the thumb

# Start the video stream
video_capture = cv2.VideoCapture(0)

while True:
    # Capture the frame from the video stream
    ret, frame = video_capture.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to the grayscale frame
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)

    # Threshold the frame to create a binary image
    _, threshold = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY)

    # Find the contours in the thresholded image
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour (the hand)
    hand_contour = max(contours, key=cv2.contourArea, default=None)

    if hand_contour is not None:
        # Draw the hand contour on the original frame
        cv2.drawContours(frame, [hand_contour], -1, (0, 255, 0), 2)

        # Get the number of raised fingers
        finger_count = find_fingers_count(hand_contour)

        # Display the finger count on the frame
        cv2.putText(frame, f"Number of Fingers: {finger_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow("Finger Count", frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video stream and close all windows
video_capture.release()
cv2.destroyAllWindows()


from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    return render_template('def_and_dumb.html')

if __name__ == '__main__':
    app.run(debug=True)



