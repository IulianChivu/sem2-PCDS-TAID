import cv2
from matplotlib import pyplot as plt
from keras.models import model_from_json
import numpy as np
import os

#load cnn model and weights
wd = os.getcwd()
json_file = open(wd + '\\model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(wd + "\\model_weights.h5")


#connect to webcam
#cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cam = cv2.VideoCapture(0)


#ask question and wait for user input
answer = input("intrebare1 ?")

#take photo
ret, frame = cam.read()
img_init = frame

#preprocess
#1 bgr to gray
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#2 resize img
frame = cv2.resize(frame, (48,48))

#3 reshape image keras [batch, w, h, channels] and feed the image to cnn input
frame_pred = frame.reshape((1, 48,48,1))
result = np.argmax(loaded_model.predict(frame_pred), axis=-1)

#4 print prediction
pred = ''
if result[0] == 0:
    pred = 'emotie prezisa: trist'
    print (pred)
if result[0] == 1:
    pred = 'emotie prezisa: neutru'
    print (pred)
if result[0] == 2:
    pred = 'emotie prezisa: fericit'
    print (pred)

#5 plot images and results
plt.figure()
plt.title("Imagine initiala. " + pred)
plt.imshow(img_init[:,:,::-1])
plt.figure()
plt.title("Imagine nivele de gri. " + pred)
plt.imshow(cv2.cvtColor(img_init, cv2.COLOR_BGR2GRAY), cmap='gray')
plt.figure()
plt.title("Imagine finala. " + pred)
plt.imshow(frame, cmap='gray')
plt.show()

cam.release()