import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
from keras.optimizers import Adam, SGD
from keras.callbacks import ReduceLROnPlateau, EarlyStopping



# Load dataset

train_data_path = 'gesture/train'
test_data_path = 'gesture/test'
train_data_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(
    directory=train_data_path, target_size=(64, 64), class_mode='categorical', batch_size=15, shuffle=True)
test_data_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(
    directory=test_data_path, target_size=(64, 64), class_mode='categorical', batch_size=15, shuffle=True)
images_train, labels_train = next(train_data_batches)
images_test, labels_test = next(test_data_batches)



# Create CNN Model 
# Convolutional Neural Network
def createModel():
    model = Sequential()
    # number of features , kernel is found all pixel in image
    model.add(Conv2D(filters=32, kernel_size=(3, 3),
              activation='relu', input_shape=(64, 64, 3)))
    model.add(MaxPool2D(pool_size=(2, 2), strides=2))

    model.add(Conv2D(filters=64, kernel_size=(3, 3),
              activation='relu', padding='same'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=2))

    model.add(Dropout(0.2))

    model.add(Conv2D(filters=128, kernel_size=(3, 3),
              activation='relu', padding='valid'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=2))

    model.add(Flatten())

    model.add(Dense(64, activation='relu'))
    model.add(Dense(128, activation='relu'))

    # for freeze and dropout some node
    model.add(Dropout(0.2))

    model.add(Dense(128, activation='relu'))

    # output layer
    model.add(Dense(11, activation='softmax'))

    return model


model = createModel()

print(model.summary())

# Training CNN Model
model.compile(optimizer=SGD(learning_rate=0.001),
              loss='categorical_crossentropy', metrics=['accuracy'])

reduceLR = ReduceLROnPlateau(
    monitor='val_loss', factor=0.2, patience=1, min_lr=0.0005)
early_stop = EarlyStopping(
    monitor='val_loss', min_delta=0, patience=1, mode='auto', verbose=0)


history = model.fit(train_data_batches, epochs=10, callbacks=[
          reduceLR, early_stop], validation_data=test_data_batches)


# Testing 
scores = model.evaluate(images_test, labels_test, verbose=0)

print(
    f'{model.metrics_names[0]} = {np.round(scores[0], 1)} || {model.metrics_names[1]} = {np.round(scores[1], 1)*100}%')

# Show 
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(loc='lower right')

# Save the model 
model.save('SignLanguageModel.hs')
