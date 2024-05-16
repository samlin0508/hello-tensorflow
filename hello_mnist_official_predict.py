import tensorflow as tf
import cv2 as cv
import numpy as np
from PIL import Image
import hello_mnist_module

model = tf.keras.models.load_model('models/hello-mnist-official.keras')

# opencv version
for i in range(0, 10):
    image = cv.imread(f'input/{i}.png', cv.IMREAD_GRAYSCALE)
    image = cv.resize(image, hello_mnist_module.mnist_image_size)
    
    image_array = np.array(image)
    image_array = 255-image_array

    hello_mnist_module.print_image(image_array)

    # image_array = image_array.reshape((1, 28, 28))
    image_array = np.array([image_array])

    predictions = model.predict(image_array)
    print(predictions)
    print(np.argmax(predictions))

# pillow version
# for i in range(0, 10):
#     image = Image.open(f'input/{i}.png').convert('L') #灰階圖片
#     image = image.resize(hello_mnist_module.mnist_image_size)

#     # 轉換格式
#     image_array = np.array(image)
#     # cv.imwrite(f'output/test2_{i}.jpg', image_array)
#     # print(image_array)

#     image_array = 255-image_array
#     cv.imwrite(f'output/{i}.jpg', image_array)

#     hello_mnist_module.print_image(image_array)

#     # image_array = np.array([image_array])
#     # image_array = image_array.reshape((1, 28, 28))
#     image_array = np.array([image_array])

#     predictions = model.predict(image_array)
#     print(predictions)
#     print(np.argmax(predictions))