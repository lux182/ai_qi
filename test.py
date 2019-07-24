from keras import models
from keras import layers
from keras import optimizers
from keras.utils import to_categorical

import numpy as np

train_images = np.array([1, 10, 20, 30, 10, 25, 30, 99, 1009])
# train_images = train_images.astype('float32') / 255
test_images = np.array([30, -30, 20, 199, 3002, 2, 25, 231, 12])
train_labels = np.array([25, 25, 25, 30, 25, 25, 30, 99, 1009])
test_labels = np.array([30, 25, 25, 199, 3002, 25, 25, 231, 25])

train_labels = np.array([1, 1, 1, 0, 1, 1, 0, 0, 0])
test_labels =  np.array([0, 1, 1, 0, 0, 1, 1, 0, 1])

model = models.Sequential()
model.add(layers.Dense(32, activation='relu', input_shape=(1 * 1,)))
# 10 路 softmax 层，它将返回一个由 10 个概率值(总和为 1)组成的数组。 每个概率值表示当前数字图像属于 10 个数字类别中某一个的概率
model.add(layers.Dense(1, activation='softmax'))

model.compile(optimizer=optimizers.RMSprop(lr=0.001), loss='mse', metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=50, batch_size=1)

test_loss, test_acc = model.evaluate(test_images, test_labels)
