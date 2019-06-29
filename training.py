import numpy as np 
import os 
import cv2 
import random 
import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten,Conv2D,MaxPooling2D

DATADIR="C:/Users/.../..." 		#path to the directory conataining the folders having the training data
CATEGORIES=["personX","personY"]

IMG_SIZE=60 

training_data=[]

def training_data():
    count=0
    for i in CATEGORIES:
        path=os.path.join(DATADIR,i)
        class_num=CATEGORIES.index(i)
        for img in os.listdir(path):
            try:
                img_array=cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
                new_array=cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
                training_data.append([new_array,class_num])
                count=count+1
                print(count)
            except Exception as e:
                pass
    
training_data()

random.shuffle(training_data)
print("shuffling done...")

X=[]
y=[]

for features,label in training_data:
    X.append(features)
    y.append(label)
	
X=np.array(X).reshape(-1,IMG_SIZE,IMG_SIZE,1)

X=X/255.0												

model=Sequential()
model.add(Conv2D(64,(3,3),input_shape=X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation("relu"))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss="binary_crossentropy",optimizer="adam",metrics=['accuracy'])

model.fit(X,y,batch_size=2,epochs=20,validation_split=0.1)

model.save("trained.model")
