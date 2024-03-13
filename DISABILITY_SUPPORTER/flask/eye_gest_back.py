import cv2
import dlib

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
