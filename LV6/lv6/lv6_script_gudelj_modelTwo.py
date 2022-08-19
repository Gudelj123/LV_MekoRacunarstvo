from keras import layers
from keras import models
from keras.datasets import mnist
from keras.utils import to_categorical

activationFunctions = ["relu", "tanh", "sigmoid" ]
sizeOfKernel = [3,5,7]
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000,28,28,1)
X_train = X_train.astype('float32') / 255
X_test = X_test.reshape(10000,28,28,1)
X_test = X_test.astype('float32') / 255
y_train = to_categorical(y_train) 
y_test = to_categorical(y_test)


for singleKernel in sizeOfKernel:
    for actFunc in range(3):

        modelTwo = models.Sequential()
        modelTwo.add(layers.Conv2D(64, (singleKernel,singleKernel), activation=activationFunctions[actFunc], input_shape=(28,28,1)))
        modelTwo.add(layers.MaxPooling2D((2, 2)))

        modelTwo.add(layers.Conv2D(32, (singleKernel,singleKernel), activation=activationFunctions[actFunc]))
        modelTwo.add(layers.MaxPooling2D((2, 2)))

        modelTwo.add(layers.Flatten())
        modelTwo.add(layers.Dense(10, activation="softmax"))

        modelTwo.compile(optimizer = "sgd",loss = "categorical_crossentropy",metrics = ["accuracy"])
        
        modelTwo.summary()
        start = time.time()
        past =  modelTwo.fit(X_train,y_train,validation_data = (X_test,y_test),epochs = 5)
        end = time.time()
        difference = end - start
        modelTwo.predict(X_test)
        #+++++KOD ZA ISPIS VRIJEDNOSTI++++++

        print("\nDONE WITH MODEL 2\n\n\ntime for training = ", str(difference), " seconds", "\nACCURACY = ", str(past.past['accuracy'][-1]))
        nameOfFile = "MODEL_2_KERNEL_SIZE=" + str(singleKernel)  + "x" + str(singleKernel) + "&activationFunc=" + activationFunctions[i] + ".txt"
        dataForWriting = "RequiredTime = " + str(difference) + " seconds" + "\nAccuracy = " + str(past.past['accuracy'][-1]) 
        with open(nameOfFile, 'w') as dat:
            dat.write(dataForWriting)        
