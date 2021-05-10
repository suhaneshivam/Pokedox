import matplotlib
matplotlib.use('Agg')

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from helper.cnn.smallvggnet import SmallVggNet
import imutils
from imutils import paths
import numpy as np
import argparse
import argparse
import os
import pickle
import random
import matplotlib.pyplot as plt
import cv2


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset (i.e., directory of images)")
ap.add_argument("-m", "--model", required=True,
	help="path to output model")
ap.add_argument("-l", "--labelbin", required=True,
	help="path to output label binarizer")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output accuracy/loss plot")
args = vars(ap.parse_args())

imagePaths = sorted(list(paths.list_images(args['dataset'])))
images = []
random.seed(42)
random.shuffle(imagePaths)
labels = []

EPOCHS = 100
INIT_LR = 1e-3
BS = 32
IMAGE_DIMS = (96, 96, 3)


for path in imagePaths:
    image = cv2.imread(path)
    image = cv2.resize(image ,(IMAGE_DIMS[1] ,IMAGE_DIMS[0]))
    image = img_to_array(image)

    label = path.split(os.path.sep)[-2]

    images.append(image)
    labels.append(label)

images = np.array(images ,dtype = "float") / 255.0
labels = np.array(labels)

(trainX ,testX ,trainY ,testY) = train_test_split(images ,labels ,test_size = 0.2 ,random_state = 42 )
aug = ImageDataGenerator(rotation_range= 25 ,width_shift_range=0.1 ,height_shift_range=0.1
                            ,shear_range=0.2 ,zoom_range=0.2,horizontal_flip = True ,fill_mode="nearest")

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

print("[INFO] compiling model...")
model = SmallVggNet.build(IMAGE_DIMS[1] ,IMAGE_DIMS[0] ,IMAGE_DIMS[2] ,classes = len(lb.classes_))

optimizer = Adam(learning_rate=0.01 ,decay=INIT_LR /EPOCHS)
model.compile(optimizer = optimizer ,loss = "categorical_crossentropy" ,metrics=["accuracy"])

print("[INFO] training network...")
H = model.fit(x = aug.flow(trainX ,trainY ,batch_size = BS) ,validation_data=(testX ,testY) ,steps_per_epoch=len(trainX) //BS ,epochs=EPOCHS ,verbose=1 )

print("[INFO] serializing network...")
model.save(args['model'] ,save_format = 'h5')

print("[INFO] serializing label binarizer...")
with open(args['labelbin'] ,'wb') as f:
    f.write(pickle.dumps(lb))

plt.style.use("ggplot")
plt.figure()
N = np.arange(0 ,EPOCHS)

plt.plot(N ,H.history["accuracy"] ,label = "Train Accuracy")
plt.plot(N ,H.history["val_accuracy"] ,label = "Validation Accuracy")
plt.plot(N ,H.history["loss"] ,label = "Train Loss")
plt.plot(N ,H.history["val_loss"] ,label = "Validation Loss")
plt.xlabel("Epochs Number")
plt.ylabel("Loss/Accuracy")
plt.legend(loc = "upper left")
plt.plot()
plt.savefig(args['plot'])
