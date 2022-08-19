from fileinput import filename
from os import times
from keras import models
from keras import layers
from keras.datasets import mnist
from tensorflow.keras.utils import to_categorical


import time


def main():

    #fetch data from mnist datset and prepare them for training
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train.reshape(60000,28,28,1)
    X_train = X_train.astype('float32')/255
    X_test = X_test.reshape(10000,28,28,1)
    X_test = X_test.astype('float32')/255
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    #create model 1

    KERNEL_SIZE = [3, 5, 7]
    ACTIVATION_FUNCTION = ['relu', 'tanh', 'sigmoid']

    EPOCH_NUMBER = 5

    for kernel in KERNEL_SIZE:
        for i in range(3):
            model =models.Sequential()
            model.add(layers.Conv2D(64, (kernel,kernel), activation=ACTIVATION_FUNCTION[i], input_shape=(28,28,1), strides=(1,1)))
            model.add(layers.Conv2D(32, (kernel,kernel), activation=ACTIVATION_FUNCTION[i], strides=(1,1)))
            model.add(layers.Flatten())
            model.add(layers.Dense(10, activation='softmax'))       
            model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

            model.summary()
            #timeStart = time.time()
            #train
            #history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs = EPOCH_NUMBER)
            #timeEnd = time.time()

            #deltaTime = timeEnd - timeStart
            #predict
            #model.predict(X_test)
            
            #print("\nDONE WITH MODEL 1\n\n\ntime for training = ", str(deltaTime), " seconds", "\nACCURACY = ", str(history.history['accuracy'][-1]))
            #fileName = "MODEL_1_KERNEL_SIZE=" + str(kernel)  + "x" + str(kernel) + "&ACTIVATION_FN=" + ACTIVATION_FUNCTION[i] + ".txt"
            #fileData = "TimeNecessary = " + str(deltaTime) + " seconds" + "\nAccuracy = " + str(history.history['accuracy'][-1]) 
            #with open(fileName, 'w') as dat:
            #    dat.write(fileData)
            


    #create model 2
    
    for kernel in KERNEL_SIZE:
        for i in range(3):


            model2 =models.Sequential()
            model2.add(layers.Conv2D(64, (kernel,kernel), activation=ACTIVATION_FUNCTION[i], input_shape=(28,28,1), strides=(1,1)))
            model2.add(layers.MaxPool2D((2,2)))
            model2.add(layers.Conv2D(32, (kernel,kernel), activation=ACTIVATION_FUNCTION[i], strides=(1,1)))
            model2.add(layers.MaxPool2D(2,2))
            model2.add(layers.Flatten())
            model2.add(layers.Dense(10, activation='softmax'))
            model2.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

            model2.summary()

            #timeStart = time.time()
            #train
            #history = model2.fit(X_train, y_train, validation_data=(X_test, y_test), epochs = EPOCH_NUMBER)
            #timeEnd = time.time()
            #deltaTime = timeEnd - timeStart
            #predict
            #model2.predict(X_test)

            #print("\nDONE WITH MODEL 2\n\n\ntime for training = ", str(deltaTime), " seconds", "\nACCURACY = ", str(history.history['accuracy'][-1]))
            #fileName = "MODEL_2_KERNEL_SIZE=" + str(kernel)  + "x" + str(kernel) + "&ACTIVATION_FN=" + ACTIVATION_FUNCTION[i] + ".txt"
            #fileData = "TimeNecessary = " + str(deltaTime) + " seconds" + "\nAccuracy = " + str(history.history['accuracy'][-1]) 
            #with open(fileName, 'w') as dat:
            #    dat.write(fileData)





if __name__ == '__main__':
    main()