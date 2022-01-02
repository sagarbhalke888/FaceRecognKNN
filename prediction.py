import pickle as pkl
import cv2
import numpy as np
import os
faceClassifier = cv2.CascadeClassifier(r'E:\\Desktop\\Sunday ML\\FaceRecognKNN\\haarcascade_frontalface_alt.xml')

vid = cv2.VideoCapture(0)

model = pkl.load(open('model.pkl', 'rb'))

while True:
    ret, frame = vid.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceClassifier.detectMultiScale(frame, 1.1, 5)
    for (x, y, w, h) in faces:
        face = gray[y:y + h, x:x + w]
        face = cv2.resize(face, (100, 100))
        arr = np.asarray(face)
        arr = arr.flatten()
        pred = model.predict(arr.reshape(1, -1))
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, pred[0], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break