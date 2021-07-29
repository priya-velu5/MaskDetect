For about the past one and a half years, coronavirus has shaken up the entire planet. The daily lifestyle has changed and the whole world
functions in a totally different manner now. In this world of uncertainties, people’s lives are in danger and must be paid more attention to.
With all the development of technology around us we as computer science engineers must put it to good use for society. Wearing a mask is
the least a person can do when going out in order to reduce risk of spread of the virus. In many public places it is hard for a human being to
manually keep track of all the visitors and check whether they are wearing a mask. Hence, we propose an automated system which can detect
whether a person is wearing a mask or not, which we have developed using deep learning algorithms.

Software Requirements

tensorflow >= 2.5.0*
keras == 2.4.3
imutils == 0.5.4
numpy == 1.19.5
opencv-python == 4.5.1.*
matplotlib == 3.4.1
argparse == 1.4.0
scipy == 1.6.2
scikit-learn == 0.24.1
pillow == 8.2.0

Hardware requirements:
1. A laptop/pc with at least 4GB RAM ( 8 GB+ preferred)
2. GPU ( for better performance)
3. Webcam
This project can be implemented for Raspberry pi too. However, the software specifications and how to run them
will not be mentioned here. This needs Raspberry Pi 4 (Pi 3 is slow for this) and PiCam module.


How to run: 
1. Download the given zip file. Unzip it. The zip file does not contain the dataset as it cannot be zipped.
2. Download the dataset from here: https://github.com/chandrikadeb7/Face-Mask-Detection/tree/master/dataset
3. Create a folder called “samples”. Create two sub-folders inside it called “inmask” and “nomask”
The directory should look like this:
samples
|-- inmask
|-- nomask
4. cd to the folder downloaded and follow the steps given. To ensure that dependency issues do not occur, create a virtual environment to run the code:
virtualenv mask
5. Activate the virtual env:
source mask/bin/activate
6. pip install -r requirements.txt

How to train

Steps to generate three different training models :
a) python training_model.py -d samples -e ResNet50V2 -m ResNet50V2.model -p ResNet50V2.png
b) python training_model.py -d samples -e MobileNetV2 -m MobileNetV2.model -p MobileNetV2.png
c) python training_model.py -d samples -e InceptionV3 -m InceptionV3.model -p InceptionV3.png
The given zip file contains all the models. This step can be skipped.

How to run the real time detector

For real time video detection with different models: (any of the given three commands can be used to run)
a) python video_scan.py -m ResNet50V2.model
b) python video_scan.py -m MobileNetV2.model
c) python video_scan.py -m InceptionV3.model


This dataset consists of 4095 images belonging to two classes:
inmask: 2165 images – Images taken of people’s faces wearing masks in different angles
nomask: 1930 images – Images taken of people’s faces without masks.
The dataset was obtained from the following source: https://github.com/chandrikadeb7/Face-Mask-Detection/tree/master/dataset
The images used were real images of faces wearing masks. The images were collected from the following sources like Bing Search API , Kaggle datasets and RMFD dataset.

reference: 
https://github.com/chandrikadeb7/Face-Mask-Detection
