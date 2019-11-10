from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator

classificador = Sequential()

classificador.add(Conv2D(32,(10,10), input_shape = (500, 500, 1), activation = 'relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size=(5,5)))

classificador.add(Conv2D(64,(10,10), activation = 'relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size=(5,5)))

classificador.add(Conv2D(128,(10,10), activation = 'relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size=(5,5)))

classificador.add(Flatten())

classificador.add(Dense(units = 1024, activation = 'relu'))#1024
classificador.add(Dropout(0.5))#0.3
classificador.add(Dense(units = 1024, activation = 'relu'))#1024
classificador.add(Dropout(0.5))#0.2

classificador.add(Dense(units = 1, activation = 'sigmoid'))

classificador.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

gerador_treinamento = ImageDataGenerator(rescale = 1./255)

gerador_validacao = ImageDataGenerator(rescale = 1./255)



base_treinamento = gerador_treinamento.flow_from_directory('dataset/training_set',
                                                           target_size = (500, 500),
                                                           batch_size = 32,
                                                           class_mode = 'binary',
                                                           color_mode = 'grayscale',
                                                           shuffle = True,
                                                           seed = 42)

base_teste = gerador_validacao.flow_from_directory('dataset/test_set', 
                                               target_size = (500, 500),
                                               batch_size = 32,
                                               class_mode = 'binary',
                                               color_mode = 'grayscale',
                                               shuffle = True,
                                               seed = 42) 

classificador.summary()

classificador.fit_generator(base_treinamento, steps_per_epoch = 100,
                            epochs = 100,#100
                            validation_data = base_teste,
                            validation_steps = 100)

classificador.summary()