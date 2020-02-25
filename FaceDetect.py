import cv2
import numpy as np
import dlib

path = "C:\\Users\\pilgrm\\Videos\\AgarMainKahoon.mp4"
subject = cv2.VideoCapture(path)

detector = dlib.get_frontal_face_detector()

# predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while True:
    _, frame = subject.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()

        # cv2.rectangle(frame, (x1,y1),(x2,y2), (0,255,0), 3)

        landmarks = predictor(gray, face)

        for n in range(68):
            x = landmarks.part(n).x 
            y = landmarks.part(n).y

            cv2.circle(frame, (x,y), 2 , (0, 0, 255), 2)
            # cv2.putText(frame, "p"+str(n),(x,y),cv2.FONT_HERSHEY_SIMPLEX ,0.3,(255,0,0),1,cv2.LINE_AA)


    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break


