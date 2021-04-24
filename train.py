import sys
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Flatten, Dense, Activation
from tensorflow.keras.layers import  Convolution2D, MaxPooling2D
from tensorflow.keras import backend as K

K.clear_session()

data_entrenamiento = './data/entrenamiento'
data_validacion = './data/validacion'
 
"""
Parameters
"""
epocas=2
longitud, altura = 150, 150
batch_size = 32
pasos = 27
validation_steps2 = 10
filtrosConv1 = 32
filtrosConv2 = 64
tamano_filtro1 = (3, 3)
tamano_filtro2 = (2, 2)
tamano_pool = (2, 2)
clases = 10
lr = 0.0004

##Preparamos nuestras imagenes

def iamgenes():
    entrenamiento_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
        )

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    entrenamiento_generador = entrenamiento_datagen.flow_from_directory(
        data_entrenamiento,
        target_size=(150, 150),
        batch_size=25,
        class_mode='categorical')

    '''img = load_img('data/entrenamiento/Excavadoras/00026.jpg')
    x = img_to_array(img)
    x = x.reshape((1,)+x.shape)

    i = 0
    for batch in entrenamiento_datagen.flow(x, batch_size=1, save_to_dir='preview', save_prefix='bus', save_format='jpeg'):
        i += 1
        if i < 20:
            break'''

    validacion_generador = test_datagen.flow_from_directory(
        data_validacion,
        target_size=(150, 150),
        batch_size=25,
        class_mode='categorical')

    return entrenamiento_generador, validacion_generador

def model():
    cnn = Sequential()
    cnn.add(Convolution2D(filters=32, kernel_size = (3,3) , activation='relu', input_shape=(longitud, altura, 3)))
    cnn.add(MaxPooling2D(2,2))

    cnn.add(Convolution2D(filters=64, kernel_size = (3,3) , activation='relu'))
    cnn.add(MaxPooling2D(2,2))

    cnn.add(Convolution2D(filters=128, kernel_size = (3,3) , activation='relu'))
    cnn.add(MaxPooling2D(2,2))

    cnn.add(Convolution2D(filters=256, kernel_size = (3,3) , activation='relu'))
    cnn.add(MaxPooling2D(2,2))

    cnn.add(Flatten())
    cnn.add(Dense(512, activation='relu'))
    cnn.add(Dropout(0.5))
    cnn.add(Dense(10, activation='softmax'))

    cnn.compile(loss='categorical_crossentropy',
                optimizer=optimizers.Adam(learning_rate = lr),
                metrics=['accuracy'])

    cnn.summary()
    return cnn

def trail_model(cnn, entrenamiento_generador, validacion_generador):
    cnn.fit(
        entrenamiento_generador,
        steps_per_epoch=27,
        epochs=50,
        validation_data=validacion_generador,
        validation_steps=19)

    target_dir = './modelo/'
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
        cnn.save('./modelo/modelo.h5')
        cnn.save_weights('./modelo/pesos.h5')

nn = model()
train, valid = iamgenes()
trail_model(nn, train, valid)