import os

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator

def fitCNN(data, target, modelName = 'model.h5', img_rows = 33, img_cols = 13):
    # input image dimension
    input_shape = (1, img_rows, img_cols)
    
    # number of classes
    nb_classes = 98
    
    # number of convolutional filters to use
    nb_filters = 32
    
    # size of pooling area for max pooling
    pool_size = (2, 2)
    
    # convolution kernel size
    kernel_size = (3, 3)
    
    # scale to [0, 1]
    X_data = data.astype('float32')
    X_data /= 255
    
    # Random image generator
    datagen = ImageDataGenerator(
        rotation_range=15,
        shear_range=0.1,
        zoom_range=0.05)
    
    # convert class vectors to binary class matrices
    Y_target = np_utils.to_categorical(target, nb_classes)
    
    # Model parameters
    batch_size = 128
    nb_filters = 32
    nb_epoch = 200
    
    # Construct model, structure similar to LeNet5 from MNIST author
    model = Sequential()
    
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                            border_mode='valid',
                            input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    
    # Adadelta is some kind of adaptive learning rate optimiser, was just in the code I copied
    model.compile(loss='categorical_crossentropy',
                  optimizer='Adadelta',
                  metrics=['accuracy'])
    
    # Model fit from data served from the generator in batches of 128, validated on another random 5000 or so 
    # observations served from generator.
    model.fit_generator(generator = datagen.flow(X_data, Y_target, batch_size=batch_size),
                        samples_per_epoch = len(X_data),
                        nb_epoch = nb_epoch,
                        validation_data = datagen.flow(X_data, Y_target, batch_size=int(len(X_data)/10)),
                        nb_val_samples = 1,
                        verbose = 1)
    
    model_path = os.path.join('Convolutional Neural Network', modelName)
    model.save(model_path)