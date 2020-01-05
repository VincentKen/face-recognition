import cv2
import numpy as np
import os
import json

recognizer = cv2.face.EigenFaceRecognizer_create()
recognizer.read("recognizer.xml")

recognizer_size = (250, 250)

with open("labels.json", 'r') as j:
    labels = json.load(j)

face_cascade = cv2.CascadeClassifier()


if not face_cascade.load(cv2.samples.findFile("haarcascade_frontalface_alt.xml")):
    print('--(!)Error loading face cascade')
    exit(0)

cap = cv2.VideoCapture(0)

font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
fontColor = (255,255,255)
lineType = 2

while True:
    ret, frame = cap.read()
    if frame is None:
        continue
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(frame_gray)
    for (x,y,w,h) in faces:
        new_img = frame_gray[y-100:y+h+100, x-100:x+w+100]
        try:
            prediction = recognizer.predict(cv2.resize(new_img, recognizer_size))
        except:
            continue
        label = labels[str(prediction[0])] + "-" + str(round(prediction[1]))
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, label, 
            (x, y), 
            font, 
            fontScale,
            fontColor,
            lineType)


    cv2.imshow('Capture - Face detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# img = cv2.resize(cv2.imread("td.jpg", 0), (250, 250))
# prediction = recognizer.predict(img)
# print(prediction)