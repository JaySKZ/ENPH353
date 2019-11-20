import math
import numpy as np
import re
import cv2
from collections import Counter
from matplotlib import pyplot as plt
from PIL import Image
import os
import random
import pickle
import sklearn
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator

from keras import layers
from keras import models
from keras import optimizers

from keras.utils import plot_model
from keras import backend

NUMBER_OF_LABELS = 26
CONFIDENCE_THRESHOLD = 0.01
LEARNING_RATE = 1e-4
VALIDATION_SPLIT = 0.15
BS = 60
EPOCHS = 30

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y

#Load Dataset
def loadDataset(PATH):
    labels = os.listdir(PATH)

    dataset = np.zeros((2*len(labels), 2), dtype=object)

    #Read labels and create y value with unicode
    for i in range(len(labels)):
        image = cv2.imread(PATH + labels[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        dataset[2*i,0] = cv2.resize(image[80:250,40:150], (64,64))
        dataset[2*i+1,0] = cv2.resize(image[80:250,150:260], (64,64))
                
        dataset[2*i,1] = ord(labels[i][6]) - 65
        dataset[2*i+1,1] = ord(labels[i][7]) - 65     

        
    print("Loaded {} images from folder".format(len(labels)))

    # Shuffle the dataset
    np.random.shuffle(dataset)

    # Split data into x and y data
    X_dataset_orig = np.array([data[0] for data in dataset[:]])
    Y_dataset_orig = np.array([[data[1]] for data in dataset]).T

    # Normalize X (images) dataset
    X_dataset = X_dataset_orig/255.
    X_dataset = np.expand_dims(X_dataset, axis=3)

    # Convert Y dataset to one-hot encoding
    Y_dataset = convert_to_one_hot(Y_dataset_orig, NUMBER_OF_LABELS).T

    #Print info about data
    print("Total examples: {:d}\nTraining examples: {:f}\nTest examples: {:f}".
        format(X_dataset.shape[0],
                math.ceil(X_dataset.shape[0] * (1-VALIDATION_SPLIT)),
                math.floor(X_dataset.shape[0] * VALIDATION_SPLIT)))
    print("X shape: " + str(X_dataset.shape))
    print("Y shape: " + str(Y_dataset.shape))


    X_dataset = np.stack(X_dataset)
    return X_dataset, Y_dataset

def trainModel(X_dataset, Y_dataset):
    conv_model = models.Sequential()
    conv_model.add(layers.Conv2D(32, (3, 3), activation='relu',
                                input_shape=(64, 64, 1)))
    conv_model.add(layers.MaxPooling2D((2, 2)))
    conv_model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    conv_model.add(layers.MaxPooling2D((2, 2)))
    conv_model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    conv_model.add(layers.MaxPooling2D((2, 2)))
    conv_model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    conv_model.add(layers.MaxPooling2D((2, 2)))
    conv_model.add(layers.Flatten())
    conv_model.add(layers.Dropout(0.5))
    conv_model.add(layers.Dense(512, activation='relu'))
    conv_model.add(layers.Dense(26, activation='softmax'))


    conv_model.compile(loss='categorical_crossentropy',
                    optimizer=optimizers.Adam(lr=1e-4),
                    metrics=['acc'])

    conv_model.summary()

    #Augment data
    (trainX, testX, trainY, testY) = train_test_split(X_dataset, Y_dataset,	test_size=VALIDATION_SPLIT)
    print(trainX.shape)
    print(testX.shape)
    aug = ImageDataGenerator(rotation_range=20,	zoom_range=0.15, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15, horizontal_flip=True, fill_mode="nearest")    
    
    # train the network

    print("[INFO] training network for {} epochs...".format(EPOCHS))
    history_conv = conv_model.fit_generator(
	aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(testX, testY),
	steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS)

    #history_conv = conv_model.fit(X_dataset, Y_dataset, 
    #                          validation_split=VALIDATION_SPLIT, 
    #                          epochs=10, 
    #                          batch_size=16)

    plt.plot(history_conv.history['loss'])
    plt.plot(history_conv.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train loss', 'val loss'], loc='upper left')
    plt.show()

    plt.plot(history_conv.history['acc'])
    plt.plot(history_conv.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy (%)')
    plt.xlabel('epoch')
    plt.legend(['train accuracy', 'val accuracy'], loc='upper left')
    plt.show()

    

    return conv_model


#Resets CNN training parameters
def reset_weights(model):
    session = backend.get_session()
    for layer in model.layers: 
        if hasattr(layer, 'kernel_initializer'):
            layer.kernel.initializer.run(session=session)
            

if __name__ == "__main__":
    x,y = loadDataset("/home/fizzer/enph353_ws/src/enph353/enph353_gazebo/media/materials/textures/dataset/")
    model = trainModel(x,y)
    pickle.dump({model}, open('lettermodel.p','wb'))
