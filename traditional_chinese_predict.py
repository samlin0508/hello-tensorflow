import tensorflow as tf
import cv2 as cv
import numpy as np
from PIL import Image
import hello_mnist_module
from os import listdir
from os.path import isfile, join

model = tf.keras.models.load_model('models/traditional-chinese.keras')
chinese_words = ['na', 'mo', 'a', 'mi', 'tuo', 'fo']

# opencv version
for i in chinese_words[0:1]:
    path = f'input/{i}'
    images = [file for file in listdir(path) if isfile(join(path, file)) and file.endswith((".png", ".jpg", "jpeg"))]

    for j in images:
        image = cv.imread(f'{path}/{j}', cv.IMREAD_GRAYSCALE)
        image = cv.resize(image, hello_mnist_module.chinese_word_image_size)
        # cv.imwrite(f'output/test_{i}.jpg', img=image)
        
        image_array = np.array(image)
        image_array = 255-image_array

        # hello_mnist_module.print_image(image_array)
        cv.imwrite(f'output/{i}_{j}', img=image)

        # image_array = image_array.reshape((1, 28, 28))
        image_array = np.array([image_array])

        predictions = model.predict(image_array)
        print(predictions)
        print(chinese_words[np.argmax(predictions)])