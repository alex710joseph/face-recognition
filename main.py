import cv2
import numpy as np
import tensorflow as tf 

CATEGORIES=["personX","personY"]

font=cv2.FONT_HERSHEY_PLAIN

face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

model=tf.keras.models.load_model("trained.model")

cap=cv2.VideoCapture(0)

while True:
	ret,frame=cap.read()
	gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	faces=face_cascade.detectMultiScale(gray,1.3,5)
	for (x,y,w,h) in faces:
		cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
		roi_gray=gray[y:y+h,x:x+w]
		roi_frame=frame[y:y+h,x:x+h]
		roi_gray=cv2.resize(roi_gray,(50,50))
		prediction=model.predict([roi_gray.reshape(-1,50,50,1)])
		cv2.putText(frame,CATEGORIES[int(prediction[0][0])],(x,y),font,1,(0,255,0),2,cv2.LINE_AA)
			
	cv2.imshow('frame',frame)
	
	if cv2.waitKey(1) & 0xff==ord('q'):
		break;
	
cap.release()
cv2.destroyAllWindows()
