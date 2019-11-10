import gc
#from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator

classificador = Sequential()

#pre processamento
#primeira camada de convolução
#                        numero de filtros(kernels), dimensao do strider, alturaXlargura, n° de canais = 1 / escala de cinza 
#classificador.add(Conv2D(64,(4,4), input_shape = (125, 125, 1), activation = 'relu'))
#classificador.add(BatchNormalization())
#classificador.add(MaxPooling2D(pool_size=(3,3)))

size = 125

classificador.add(Conv2D(32, (3, 3), padding='same',input_shape=(size,size,1), activation = 'relu'))
classificador.add(BatchNormalization())
classificador.add(Conv2D(32, (3, 3), activation = 'relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D())
classificador.add(Dropout(0.25))

#classificador.add(MaxPooling2D(pool_size=(2, 2)))
#classificador.add(Dropout(0.25))

#classificador.add(Conv2D(32, (3, 3), activation = 'relu'))
#classificador.add(BatchNormalization())
#classificador.add(MaxPooling2D(pool_size=(2, 2)))
#classificador.add(Dropout(0.25))

#segunda camada de convolução
#classificador.add(Conv2D(64,(4,4), input_shape = (125, 125, 1), activation = 'relu'))
#classificador.add(BatchNormalization())
#classificador.add(MaxPooling2D(pool_size = (3,3)))

#flattening
classificador.add(Flatten())

#rede neural densa
#camada oculta
#                       numero de neuronios
#classificador.add(Dense(units = 128, activation = 'relu'))
#                neuronios descartados
#classificador.add(Dropout(0.2))
#                       numero de neuronios
classificador.add(Dense(units = 128, activation = 'relu'))
#                neuronios descartados
classificador.add(Dropout(0.2))
#               camada de saida = 1, ou tem cancer ou nao tem cancer
classificador.add(Dense(units = 1, activation = 'sigmoid'))

classificador.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#normalizando o vetor e aumentando a quantidade de imagens
gerador_treinamento = ImageDataGenerator(rescale = 1./255, 
                                         rotation_range = 7, 
                                         horizontal_flip = True, 
                                         shear_range = 0.2, 
                                         height_shift_range = 0.07, 
                                         zoom_range = 0.2)
#normalizando o vetor
gerador_teste = ImageDataGenerator(rescale = 1./255)


#carregando as imagens para treinamento 91 total, e convertendo para gray scale
base_treinamento = gerador_treinamento.flow_from_directory('dataset/training_set',
                                                           target_size = (size, size),
                                                           batch_size = 32,
                                                           class_mode = 'binary',
                                                           color_mode = 'grayscale') #'dataset/treinamento_set''C:/Users/BlueTerror/Desktop/CNN_Cancer/dataset/treinamento_set'
 
#carregando as imagens para teste 60 no total, e convertendo para gray scale
base_teste = gerador_teste.flow_from_directory('dataset/test_set', 
                                               target_size = (size, size),
                                               batch_size = 32,
                                               class_mode = 'binary',
                                               color_mode = 'grayscale') #'dataset/teste_set''C:/Users/BlueTerror/Desktop/CNN_Cancer/dataset/teste_set'

#                                                   numero de imagens na base de treinamento
classificador.summary()
classificador.fit_generator(base_treinamento, steps_per_epoch = 50,
                            epochs = 25,
                            validation_data = base_teste,
                            validation_steps = 50)