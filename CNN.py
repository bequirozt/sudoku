from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam

## Carga el dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

## Redimenciona la data
X_train = X_train.reshape((X_train.shape[0],28,28,1))
X_test  = X_test.reshape((X_test.shape[0],28,28,1))

## Normaliza la data
X_train = X_train.astype('float32')/255.0
X_test  = X_test.astype('float32')/255.0

## Transforma los labels de numeros a arreglos
y_train = y_train.reshape(len(y_train),1)
y_train = OneHotEncoder(sparse=False).fit_transform(y_train)
y_test = y_test.reshape(len(y_test),1)
y_test = OneHotEncoder(sparse=False).fit_transform(y_test)

## Creación de la red neuronal convolucional
CNN = Sequential()
CNN.add(Conv2D(32,(5,5), input_shape=(28,28,1), 
        padding='same',activation='relu'))
CNN.add(MaxPooling2D(pool_size=(2,2)))

CNN.add(Conv2D(32,(3,3), padding='same',activation='relu'))
CNN.add(MaxPooling2D(pool_size=(2,2)))

CNN.add(Flatten())
CNN.add(Dense(64, activation='relu'))
CNN.add(Dropout(0.5))

CNN.add(Dense(64, activation='relu'))
CNN.add(Dropout(0.5))

CNN.add(Dense(10, activation='softmax'))

## Compilación y entrenamiento
CNN.compile(loss='categorical_crossentropy',
            optimizer=Adam(lr=1e-3),
            metrics=['accuracy'])
CNN.fit(X_train, y_train,
        validation_data=(X_test,y_test),
        epochs=10, batch_size=128)

## Evaluar la CNN
score = CNN.evaluate(X_test, y_test, verbose=0)


## Guardar el modelo
model_jason = CNN.to_json()

with open('CNN.json', 'w') as json_file:
    json_file.write(model_jason)

CNN.save_weights('model.h5')



