from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Load dataset
train_data_path = 'gesture/train'
test_data_path = 'gesture/test'
train_data_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(
    directory=train_data_path, target_size=(64, 64), class_mode='categorical', batch_size=15, shuffle=True)
test_data_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(
    directory=test_data_path, target_size=(64, 64), class_mode='categorical', batch_size=15, shuffle=True)
images_train, labels_train = next(train_data_batches)
images_test, labels_test = next(test_data_batches)


def display(images):
    fig, axes = plt.subplots(1, 15, figsize=(50, 50))
    for image, ax in zip(images, axes):
        image = np.clip(image, 0, 1)
        ax.imshow(image)
        ax.axis('off')
    plt.tight_layout
    plt.show()


words = {0: 'Zero', 1: 'One', 2: 'Two', 3: 'Three', 4: 'Four',
         5: 'Five', 6: 'Six', 7: 'Seven', 8: 'Eight', 9: 'Nine', 10: 'Ten'}

display(images_test)
labels_test


def digits(labels):
    for label in labels:
        print(words[np.argmax(label)], end='  ')


digits(labels_test)

print('\n', images_test[0].shape)
