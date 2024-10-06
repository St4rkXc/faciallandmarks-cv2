import cv2
import mediapipe as mp

# face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# hand drawing
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mphands = mp.solutions.hands
hand = mphands.Hands()
open_cap = cv2.VideoCapture(0)

while True:
    ret, image = open_cap.read()
    height, width, _ = image.shape
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # facial landmarks
    resultface = face_mesh.process(rgb_image)
    if resultface.multi_face_landmarks:
        for facial_landmarks in resultface.multi_face_landmarks:
            for i in range(0, 468):
                pt1 = facial_landmarks.landmark[i]
                x = int(pt1.x * width)
                y = int(pt1.y * height)
                cv2.circle(image, (x, y), 2, (100, 0, 0), -1)
                # cv2.putText(image, str(i), (x,y),0,1,(0,0,0))
    
    # hand landmarks
    resulthand = hand.process(image)
    if resulthand.multi_hand_landmarks:
        for hand_landmark in resulthand.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmark, mphands.HAND_CONNECTIONS)
    
    cv2.imshow("Image", image)
    key = cv2.waitKey(1)
    if key == 27:
        break
