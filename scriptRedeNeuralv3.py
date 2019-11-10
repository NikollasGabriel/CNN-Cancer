from PIL import Image, ImageOps
import numpy as np
import os
from sklearn import preprocessing 
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

imagens = []
path = 'C:/Users/BlueTerror/Desktop/CNN_Cancer/Base/'

for folder in os.listdir(path):
     for file in os.listdir(os.path.join(path, folder)):
         desired_size = 128
         im = Image.open(path + folder + '/' + file)
         old_size = im.size  # old_size[0] is in (width, height) format
         ratio = float(desired_size)/max(old_size)
         new_size = tuple([int(x*ratio) for x in old_size])
         im = im.resize(new_size, Image.ANTIALIAS)
         # create a new image and paste the resized on it
         new_im = Image.new("L", (desired_size, desired_size), (255))
         new_im.paste(im, ((desired_size-new_size[0])//2,(desired_size-new_size[1])//2))
         new_im = ImageOps.equalize(new_im)
         imagens.append(new_im)
         
imagens = np.array(imagens, dtype='float') / 255.0

#First, Convert to int labels
labels = preprocessing.LabelEncoder()
train_vals = np.array(labels)
int_encoded = labels.fit_transform(train_vals)

#reshape to prepare for one hot encoding
reshape_intEncoded = int_encoded.reshape(-1,1)

X_train, X_valid, y_train, y_valid = train_test_split(imagens, reshape_intEncoded, test_size=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

y_train = to_categorical(y_train, num_classes=2)
y_valid = to_categorical(y_valid, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)