# This is core file


import numpy as np
import cv2
import pickle

face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
#eye_cascade = cv2.CascadeClassifier('data/haarcascade_eye.xml')
#smile_cascade = cv2.CascadeClassifier('data/haarcascade_smile.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")

labels = {"person_name": 1}
with open("labels.pickle", "rb") as f:
	og_labels = pickle.load(f)
	labels = {v:k for k,v in og_labels.items()}

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    # Our operations on the frame come here

    for (x,y,w,h) in faces:
    	#print(x,y,w,h)
    	roi_gray = gray[y:y+h, x:x+w]
    	roi_frame = frame[y:y+h, x:x+w]
    	
    	# recognize
    	id, conf = recognizer.predict(roi_gray)
    	if conf>=45: 
    		#print(id)
    		print(labels[id])
    		print(conf)
    		font = cv2.FONT_HERSHEY_SIMPLEX
    		name = labels[id]
    		color = (0,255,0)
    		stroke = 2
    		cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)
    	img_item = "{}-{}.png".format(name, id)
    	cv2.imwrite(img_item, roi_frame)

    	color = (255, 255, 255) #BGR
    	stroke = 2
    	width = w + x
    	height = h + y
    	cv2.rectangle(frame, (x, y), (width, height), color, stroke)
    	#eyes = eye_cascade.detectMultiScale(roi_gray)
    	#for (ex, ey, ew, eh) in eyes:
    		#cv2.rectangle(roi_frame, (ex, eh), (ex+ew, ey+eh), (0,0,255),1)
    	#smile = smile_cascade.detectMultiScale(roi_gray)
    	#for (sx, sy, sw, sh) in smile:
    		#cv2.rectangle(roi_frame, (sx, sh), (sx+sw, sy+sh), (0,0,255),1)

    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()