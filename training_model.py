# Usage: python training_model.py -d <<directory>> -m <<model output file>> -e <<Neural engine>>
# Training model is to train the model from existing data sets
import argparse
# A series of convenience functions to make basic image processing functions such as translation, 
# rotation, resizing, skeletonization, displaying Matplotlib images, sorting contours, detecting edges
from imutils import paths
from tensorflow.keras.preprocessing.image import load_img,img_to_array,ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
# NumPy is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices
import numpy as np
# LabelBinarizer Binarize labels in a one-vs-all fashion
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import MobileNetV2,ResNet50V2
from tensorflow.keras.applications.inception_v3 import InceptionV3

from tensorflow.keras.layers import AveragePooling2D,Flatten, Dense,Dropout,Input #layers of the CNN
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam,RMSprop
import matplotlib.pyplot as plt
import os
# 
IMG_SIZE = 224
CHANNELS = 3 #rgb images 
# argparse — Parser for command-line options, arguments and sub-commands
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--directory", required=True,	help="directory path to input images")
ap.add_argument("-p", "--plot", type=str, default="output.png", help="path to output result - loss/accuracy")
ap.add_argument("-m", "--model", type=str,	default="faceMask.model", 	help="path to output face mask model")
ap.add_argument("-e", "--engine", type=str,	default="MobileNetV2", 	help="Keras Neural applications")
args = vars(ap.parse_args())
print("[DEBUG] training images...")
imagePaths = list(paths.list_images(args["directory"]))
data = []
labels = []

# Start Loops
for images in imagePaths:
	# extract the label value from file name
	label = images.split(os.path.sep)[-2]
	#load image and set the size 
	image = load_img(images, target_size=(IMG_SIZE, IMG_SIZE))
	#Converts a PIL Image instance to a Numpy array
	image = img_to_array(image)
	#preprocess the image using keras api for preprocessing 
	image = preprocess_input(image)
	#maintain lists of images and their corresponding labels 
	data.append(image)
	labels.append(label)

# convert the data and labels to NumPy arrays. 
''' 
Numpy array is densely packed in memory due to its homogeneous type, it also frees the memory faster.
So overall a task executed in Numpy is around 5 to 100 times faster than the standard python list,
which is a significant leap in terms of speed.
'''
data = np.array(data, dtype="float32")
labels = np.array(labels)
# perform one-hot encoding on the labels 
#this is used to convert multiclass labels to binary labels(i.e, belongs or doesnt belong to a class)
lb = LabelBinarizer() #this makes an object to do the above
labels = lb.fit_transform(labels) # the above mentioned action takes place here 
#Converts a class vector (integers) to binary class matrix
labels = to_categorical(labels)
# 20% data in test sets,80% for training
(trainX, testX, trainY, testY) = train_test_split(data, labels,test_size=0.20, stratify=labels, random_state=42)

#  the training image generator for images
aug = ImageDataGenerator( rotation_range=20, zoom_range=0.15,	width_shift_range=0.2,height_shift_range=0.2, 	
shear_range=0.15, horizontal_flip=True, fill_mode="nearest")

#choosing the model to train 
''' 
imagenet - dataset used to pretrain these models 
include_top - Boolean, whether to include the fully-connected
      layer at the top of the network. Defaults to `True`.
'''
if (args["engine"] == "MobileNetV2") :
   baseModel = MobileNetV2(weights="imagenet", include_top=False,input_tensor=Input(shape=(IMG_SIZE, IMG_SIZE, CHANNELS)))
elif (args["engine"] == "ResNet50V2") :
   baseModel = ResNet50V2(weights="imagenet", include_top=False,input_tensor=Input(shape=(IMG_SIZE, IMG_SIZE, CHANNELS)))
elif (args["engine"] == "InceptionV3") :
   baseModel = InceptionV3(weights="imagenet", include_top=False,input_tensor=Input(shape=(IMG_SIZE, IMG_SIZE, CHANNELS)))

# construct the head of the model that will be placed on top of the
# the base model
'''
Flatten is the function that converts the pooled feature map to a single column
that is passed to the fully connected layer.

Dense adds the fully connected layer to the neural network.

Dropout helps to prevent overfitting.The Dropout layer randomly sets input units to 0 with a 
frequency of `rate` at each step during training time. 

Pooling is basically “downscaling” the image obtained from the previous layers.
Average pooling - Downsamples the input along its spatial dimensions (height and width)
  by taking the average value over an input window
  (of size defined by `pool_size`) for each channel of the input.

Activation function - It is used to determine the output of neural network like yes or no.
ReLU (Rectified Linear Unit) 

softmax function is a more generalized logistic activation function which
is used for multiclass classification.
'''
headModel = baseModel.output
if (args["engine"] == "InceptionV3") :
	headModel = Flatten(name="flatten")(headModel)
	headModel = Dense(128, activation="relu")(headModel)
	headModel = Dropout(0.5)(headModel)
	headModel = Dense(2, activation="softmax")(headModel)
else :
	headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
	headModel = Flatten(name="flatten")(headModel)
	headModel = Dense(128, activation="relu")(headModel)
	headModel = Dropout(0.5)(headModel)
	headModel = Dense(2, activation="softmax")(headModel)

# place the head FC model on top of the base model (this will become the actual model we will train)
model = Model(inputs=baseModel.input, outputs=headModel)
# initialize the initial learning rate, number of epochs to train for,and batch size
#The learning rate controls how quickly the model is adapted to the problem.
INIT_LR = 1e-4
EPOCHS = 7
BS = 32
# loop over all layers in the base model and freeze them so they will
for layer in baseModel.layers:
	layer.trainable = False
# compile our model
print("[INFO] compiling model...")
if (args["engine"] == "InceptionV3") :
  opt = RMSprop(lr=INIT_LR)
else :
   opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)


model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# train the head of the network
print("[INFO] training head...")
H = model.fit(
	aug.flow(trainX, trainY, batch_size=BS),
	steps_per_epoch=len(trainX) // BS,
	validation_data=(testX, testY),
	validation_steps=len(testX) // BS,
	epochs=EPOCHS)

# make predictions on the testing set
print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=BS)

# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
predIdxs = np.argmax(predIdxs, axis=1)

# show a nicely formatted classification report
print(classification_report(testY.argmax(axis=1), predIdxs,	target_names=lb.classes_))

# serialize the model to disk
print("[INFO] saving mask detector model...")
model.save(args["model"], save_format="h5")

# plot the training loss and accuracy
N = EPOCHS
plt.style.use("ggplot")
plt.rcParams['lines.linewidth']=3
plt.rcParams['axes.facecolor']='y'
plt.rcParams['xtick.color']='r'
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title(args["engine"])
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy #")
plt.legend(loc="lower left")
plt.savefig(args["plot"])