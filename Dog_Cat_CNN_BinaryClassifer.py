import tensorflow as tf
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Flatten,Dense,Conv2D,MaxPool2D,Activation   
import matplotlib.pyplot as plt
import os 
import cv2
import random 
import numpy as np  

dir_img="PetImages" 
categories=["Cat","Dog"] 
data_training=[] 
img_size=50
def create_training():
     for category in categories:
            path=os.path.join(dir_img,category) 
            target_num=categories.index(category)
            for img in os.listdir(path):
                try:
                    img_array=cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE) 
                    new_array=cv2.resize(img_array,(img_size,img_size))  
                    data_training.append([new_array,target_num]) 
                except Exception as e:
                    pass

create_training()

random.shuffle(data_training)
x=[]
y=[]
for image,target in data_training:
    x.append(image) 
    y.append(target)
x_arr=np.array(x).reshape(-1,img_size,img_size,1) 
y_arr=np.array(y)
x=x_arr
y=y_arr

x=x/255
model=Sequential() 
model.add(Conv2D(64,(3,3),input_shape=x.shape[1:])) 
model.add(Activation("relu")) 
model.add(MaxPool2D(pool_size=(2,2))) 

model.add(Conv2D(64,(3,3)))  
model.add(Activation("relu")) 
model.add(MaxPool2D(pool_size=(2,2))) 
 

model.add(Conv2D(64,(3,3))) 
model.add(Activation("relu"))
model.add(MaxPool2D(pool_size=(2,2)))    
    
model.add(Flatten()) 

model.add(Dense(1)) 
model.add(Activation("sigmoid"))  

model.compile(loss="binary_crossentropy",optimizer="adam",metrics=["accuracy"])  

model.fit(x,y,batch_size=64,validation_split=0.1,epochs=15)




model.save("cat_dog_CNN_recognizer.model") 

model_cnn=tf.keras.models.load_model("cat_dog_CNN_recognizer.model")
categories_pets=["cats","dogs"]
def create_testing(dir_img):
    testing=[]
    for category in categories_pets: 
        num_class=categories_pets.index(category) 
        path=os.path.join(dir_img,category)
        try:
            for img in os.listdir(path):
                img_arr=cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE) 
                img_new=cv2.resize(img_arr,(img_size,img_size)) 
                img_new=img_new.reshape(-1,img_size,img_size,1)
                testing.append([img_new,num_class])
        except Exception as e: 
            pass
    return testing
test_arr=create_testing("test_set")
test_arr=np.array(test_arr) 
random.shuffle(test_arr)
x_test=[] 
y_test=[]
for x,y in test_arr: 
    x_test.append(x)
    y_test.append(y) 
x_test=np.array(x_test)
y_test