from keras.models import Sequential
from keras.layers import RandomFlip, RandomRotation, RandomZoom
import tensorflow as tf
import numpy as np
import cv2
import os
import glob as gb

# Convert Labels
code = {'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
        'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10}


def get_code(num):
    for label, index in code.items():
        if num == index:
            return label


# Load dataset
x_test = []
y_test = []

for folder in os.listdir('C:\\Users\\Mahmoud\\Downloads\\Compressed\\Sign Language\\gesture\\train'):
    files = gb.glob(pathname=str(
        'C:\\Users\\Mahmoud\\Downloads\\Compressed\\Sign Language\\gesture\\train\\' + folder + '\*.jpg'))
    print(folder)
    for file in files:
        image = cv2.imread(file)
        image_array = cv2.resize(image, (64, 64))
        x_test.append(list(image_array))
        y_test.append(int(folder))


# Increase dataset function
data_augmentation = Sequential([
    # RandomFlip((150, 150, 1)),
    RandomRotation(0.1),
    RandomZoom(0.1)
])


for folder in os.listdir('C:\\Users\\Mahmoud\\Downloads\\Compressed\\Sign Language\\gesture\\train'):
    files = gb.glob(pathname=str(
        'C:\\Users\\Mahmoud\\Downloads\\Compressed\\Sign Language\\gesture\\train\\' + folder + '\*.jpg'))
    index = 1
    for file in files:
        image = cv2.imread(file)
        image_s = cv2.resize(image, (64, 64))
        for x in range(5):
            new_image = data_augmentation(tf.expand_dims(image_s, 0))
            new_image = np.array(new_image, dtype='float32')
            cv2.imwrite('C:\\Users\\Mahmoud\\Downloads\\Compressed\\Sign Language\\gesture\\train\\' + folder + '\\' +
                        'augmentation' + '-' + str(index) + '-' + str(x+1) + '.jpg', new_image[0])
            print(f'{folder} : {index} / {len(files)}')
        index += 1
