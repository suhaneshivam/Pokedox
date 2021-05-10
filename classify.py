from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import argparse
import os
import pickle
from sklearn.metrics import classification_report
import imutils

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained model model")
ap.add_argument("-l", "--labelbin", required=True,
	help="path to label binarizer")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())

model = load_model(args['model'])
lb = pickle.loads(open(args['labelbin'] ,"rb").read())


image = cv2.imread(args["image"])
output = image.copy()

image = cv2.resize(image ,(96 ,96))
image = image.astype("float") /255.0
image = img_to_array(image)
#or we can use image = image[np.newaxis]
image = np.expand_dims(image ,axis = 0)

predictions = model.predict(image)
#print(predictions)
idx = np.argmax(predictions ,axis = 1)
#print(idx)
label = lb.classes_[idx[0]]

filename = args["image"][args["image"].rfind(os.path.sep) +1 :]
correct = "correct" if filename.rfind(str(label)) != -1 else "incorrect"

label = "{} :{:.2f}% ({})".format(str(label) ,(predictions[0][idx[0]]*100) ,correct)

print("[INFO] :{} ".format(label))
output = imutils.resize(output, width=400)
cv2.putText(output ,label ,(10 ,30) ,cv2.FONT_HERSHEY_SIMPLEX ,0.7 ,(0 ,0 ,255) ,2)
cv2.imshow("Pokemon" ,output)
cv2.waitKey(0)
