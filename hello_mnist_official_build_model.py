import tensorflow as tf
import cv2 as cv
import numpy as np
from skimage import io
from skimage.transform import resize
from PIL import Image
import hello_mnist_module

mnist = tf.keras.datasets.mnist

# 匯入 MNIST 手寫阿拉伯數字 訓練資料
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# cv.imwrite('output/train_0.jpg', x_train[0])
# print(x_train[0])

hello_mnist_module.print_image(x_train[0])

# 建立模型
model = tf.keras.models.Sequential([
    # 特徵縮放，使用常態化(Normalization)，公式 = (x - min) / (max - min)
    # 顏色範圍：0~255，所以，公式簡化為 x / 255
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Flatten(input_shape=hello_mnist_module.mnist_image_size),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
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
model.evaluate(x_test, y_test)

# # 實際預測 20 筆
# predictions = model.predict(x_test[0:20])
# # get prediction result
# print('prediction:', predictions[0:20])
# print('actual    :', y_test[0:20])

model.save('models/hello-mnist-official.keras')