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

NUMBER_OF_LABELS = 8
CONFIDENCE_THRESHOLD = 0.01
LEARNING_RATE = 1e-5
VALIDATION_SPLIT = 0.15
BS = 32
EPOCHS = 30

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y

def leftVal(val):
    return (cv2.boundingRect(val))[0]

#Load Dataset
def loadDataset(PATH):
    labels = os.listdir(PATH)

    dataset = np.zeros((len(labels), 2), dtype=object)

    #Read labels and create y value with unicode
    for i in range(len(labels)):
        image = cv2.imread(PATH + labels[i])
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh, image = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours.sort(key=leftVal, reverse=False)

        x,y,w,h = cv2.boundingRect(contours[1])
        dataset[i,0] = cv2.resize(image[y-30:y+h+30, x-30:x+w+30], (64,64))
        dataset[i,1] = ord((labels[i][0])) - 49

    print("Loaded {} images from folder".format(len(labels)))

    #Randomly Augment Data
    #for i in range(2*len(labels)):
        #cv2.imshow('y',dataset[i,0])
        #cv2.waitKey(0)

    # Split data into x and y data
    X_dataset_orig = np.array([data[0] for data in dataset[:]])
    Y_dataset_orig = np.array([[data[1]] for data in dataset]).T

    # Normalize X (images) dataset
    X_dataset = X_dataset_orig
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
    conv_model.add(layers.Dense(8, activation='softmax'))


    conv_model.compile(loss='categorical_crossentropy',
                    optimizer=optimizers.Adam(lr=LEARNING_RATE),
                    metrics=['acc'])

    conv_model.summary()

    #Augment data
    (trainX, testX, trainY, testY) = train_test_split(X_dataset, Y_dataset,	test_size=VALIDATION_SPLIT)
    aug = ImageDataGenerator(horizontal_flip=False, shear_range=random.uniform(2,5), width_shift_range=4, height_shift_range=4)     
    #for x,y in aug.flow(trainX, trainY):
    #    for i in range(len(x)):
    #        cv2.imshow('window', x[i])
    #        print(y[i])
    #        cv2.waitKey(0)
    # train the network

    print("[INFO] training network for {} epochs...".format(EPOCHS))
    history_conv = conv_model.fit_generator(
	aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(aug.flow(testX, testY, batch_size=BS)),
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
    #plt.show()

    plt.plot(history_conv.history['acc'])
    plt.plot(history_conv.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy (%)')
    plt.xlabel('epoch')
    plt.legend(['train accuracy', 'val accuracy'], loc='upper left')
    #plt.show()

    

    return conv_model


#Resets CNN training parameters
def reset_weights(model):
    session = backend.get_session()
    for layer in model.layers: 
        if hasattr(layer, 'kernel_initializer'):
            layer.kernel.initializer.run(session=session)
            

if __name__ == "__main__":
    x,y = loadDataset("/home/fizzer/enph353_ws/src/enph353/enph353_gazebo/media/materials/textures/dataset2/")
    model = trainModel(x,y)
    model.save("parkingmodel.h5")