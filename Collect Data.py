import cv2
import numpy as np
import os

vid = cv2.VideoCapture(0)
name = input("Enter your name: ")
faceClassifier = cv2.CascadeClassifier(r'E:\\Desktop\\Sunday ML\\FaceRecognKNN\\haarcascade_frontalface_alt.xml')
i = 0
while i < 200:
    ret, frame = vid.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceClassifier.detectMultiScale(frame, 1.1, 5)
    for (x, y, w, h) in faces:
        face = gray[y:y + h, x:x + w]
        face = cv2.resize(face, (100, 100))
        arr = np.asarray(face)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.imwrite(os.path.join("Data",name+"_"+str(i)+".jpg"), face)

    i += 1
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('a'):
        break

vid.release()
cv2.destroyAllWindows()
