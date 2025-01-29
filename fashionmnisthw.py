import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.datasets import fashion_mnist 
import matplotlib.pyplot as plt


mnist  = tf.keras.datasets.fashion_mnist
#load data 
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

#checking shapes
print(f"training data shape: {X_train.shape}")
print(f"test data shape: {X_test.shape}")

# Reshape and normalize the images
X_train = X_train.reshape(60000, 28, 28, 1).astype('float32') / 255
X_test = X_test.reshape(10000, 28, 28, 1).astype('float32') / 255



#build neural net
model = tf.keras.Sequential([
    #convulational layer - feature extraction
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1), 
                  kernel_regularizer=regularizers.l2(0.001)),  # L2 Regularization
    layers.BatchNormalization(),
    

    #second layer
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),
    layers.Dropout(0.5), #to reduce overfitting
    #third layer
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)), 
    layers.Dropout(0.6), #to reduce overfitting


    # flatten input image to 1D 
    layers.Flatten(), 

    #dense layer and ReLU activation
    layers.Dense(256, activation = 'relu'), 
    layers.Dropout(0.5), #prevents overfitting

    #fourth layer, output layer
    layers.Dense(10, activation = 'softmax')
])


#model compilation using Adam
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

#train the model 
history_adam = model.fit(X_train, y_train, batch_size=64, epochs=40, validation_data=(X_test, y_test), verbose = 1) 

#test accuracy 
test_loss_adam, test_accuracy_adam = model.evaluate(X_test, y_test)
print(f"Test Accuracy using Adam optimizer: {test_accuracy_adam}") 


# compiling using SGD optimizer
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history_SGD = model.fit(X_train, y_train, batch_size=32, epochs=10, verbose=1, validation_data=(X_test, y_test))
test_loss_SGD, test_accuracy_SGD = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy_SGD}")




#model compilation using rmsprop
model.compile(optimizer = 'rmsprop', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

#train model
history_rmsprop = model.fit(X_train, y_train, batch_size=128, epochs=20, verbose=1, validation_data=(X_test, y_test))
history = history_rmsprop.history 
#test accuracy
test_loss_rmsprop, test_accuracy_rmsprop = model.evaluate(X_test, y_test)
print(f"Test Accuracy using Rmsprop: {test_accuracy_rmsprop}")

plt.figure(figsize=(12, 6))
# Accuracy Plot
plt.subplot(1, 2, 1)
plt.plot(history['accuracy'], label='Training Accuracy')
plt.plot(history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)


plt.savefig("accuracy.png")
plt.show()