import tensorflow as tf
import cv2 as cv
import numpy as np
import hello_mnist_module
from os import listdir
from os.path import isfile, join

chinese_words = ['na', 'mo', 'a', 'mi', 'tuo', 'fo']
x_train = []
y_train = []

for i in chinese_words:
    path = f'training-data/{i}'
    images = [file for file in listdir(path) if isfile(join(path, file)) and file.endswith((".png", ".jpg", "jpeg"))]

    for j in images:
        image = cv.imread(f'{path}/{j}', cv.IMREAD_GRAYSCALE)
        image = cv.resize(image, hello_mnist_module.chinese_word_image_size)
        
        image_array = np.array(image)
        image_array = 255-image_array

        # hello_mnist_module.print_image(image_array)

        x_train.append(image_array)
        y_train.append(chinese_words.index(i))

x_train = np.array(x_train)
y_train = np.array(y_train)
# hello_mnist_module.print_image(x_train[0])    
# print(x_train[0])

# 建立模型
model = tf.keras.models.Sequential([
    # 特徵縮放，使用常態化(Normalization)，公式 = (x - min) / (max - min)
    # 顏色範圍：0~255，所以，公式簡化為 x / 255
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Flatten(input_shape=hello_mnist_module.chinese_word_image_size),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(6, activation='softmax')
])

# 設定優化器(optimizer)、損失函數(loss)、效能衡量指標(metrics)的類別
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 模型訓練
history = model.fit(x_train, y_train, epochs=5)

# 模型評估，打分數
model.evaluate(x_train, y_train)

model.save('models/traditional-chinese.keras')