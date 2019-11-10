import os
import random
import gc
import cv2
import numpy as np
from tensorflow.keras import backend as k
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
#from keras.layers.normalization import BatchNormalization
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array, load_img

cancer_benigno_path = 'C:/Users/BlueTerror/Desktop/CNN_Cancer/Base/benign'#'C:/Users/BlueTerror/Desktop/CNN_Cancer/dataset/benign''C:/Users/BlueTerror/Desktop/CNN_Cancer/dataset/benign/{}'
cancer_maligno_path = 'C:/Users/BlueTerror/Desktop/CNN_Cancer/Base/malignant'#'C:/Users/BlueTerror/Desktop/CNN_Cancer/dataset/malignant''C:/Users/BlueTerror/Desktop/CNN_Cancer/dataset/malignant/{}'

cancer_benigno_pasta = ['C:/Users/BlueTerror/Desktop/CNN_Cancer/Base/benign/{}'.format(i) for i in os.listdir(cancer_benigno_path) if 'benign' in i]
cancer_maligno_pasta = ['C:/Users/BlueTerror/Desktop/CNN_Cancer/Base/malignant/{}'.format(i) for i in os.listdir(cancer_maligno_path) if 'malignant' in i]

uniao_dados = cancer_benigno_pasta + cancer_maligno_pasta

random.shuffle(uniao_dados)

del cancer_benigno_pasta
del cancer_maligno_pasta
del cancer_benigno_path
del cancer_maligno_path
gc.collect()

nrows = 125
ncolumns = 125
channels = 1

def read_and_process_image(list_of_images):
    X = []
    y = []
    #(cv2.imread(image,cv2.IMREAD_GRAYSCALE), (nrows,ncolumns), interpolation=cv2.INTER_CUBIC)
    for image in list_of_images:
        X.append(cv2.resize(cv2.imread(image,cv2.IMREAD_COLOR), (nrows,ncolumns), interpolation=cv2.INTER_CUBIC))
        if 'benign' in image:
            y.append(1)
        elif 'malignant' in image:
            y.append(0)
    return X,y

X,y = read_and_process_image(uniao_dados) 

X = np.array(X)
y = np.array(y)

X_train, X_val, y_train, y_val  = train_test_split(X, y, test_size=0.30, random_state=2)

X_train = np.array(X_train)
X_val = np.array(X_val)
"""del X
del y
gc.collect()"""

ntrain = len(X_train)
nval = len(X_val)
batch_size = 32

classificador = Sequential()

classificador.add(Conv2D(32,(3,3), input_shape = (125, 125, 1), activation = 'relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size=(2,2)))

classificador.add(Conv2D(64,(3,3), activation = 'relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size=(2,2)))

classificador.add(Conv2D(128,(3,3), activation = 'relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size=(2,2)))

classificador.add(Conv2D(128,(3,3), activation = 'relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size=(2,2)))

classificador.add(Flatten())

classificador.add(Dropout(0.5))
classificador.add(Dense(units = 512, activation = 'relu'))

classificador.add(Dense(units = 1, activation = 'sigmoid'))

classificador.summary()

classificador.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

gerador_treinamento = ImageDataGenerator(rescale = 1./255, 
                                         rotation_range = 40, 
                                         horizontal_flip = True, 
                                         shear_range = 0.2, 
                                         width_shift_range = 0.02,
                                         height_shift_range = 0.02, 
                                         zoom_range = 0.2)

gerador_validacao = ImageDataGenerator(rescale = 1./255)

base_treinamento = gerador_treinamento.flow(X_train, y_train, batch_size = 32)
base_validacao = gerador_validacao.flow(X_val, y_val, batch_size = 32)

classificador.fit(gerador_treinamento, steps_per_epoch= ntrain // batch_size,
                                      validation_data = gerador_validacao,
                                      validation_steps = nval // batch_size)

#classificador.summary()"""