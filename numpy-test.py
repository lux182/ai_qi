from keras.datasets import mnist
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 轴的个数
# 3
print(train_images.ndim)

# 形状
# (60000, 28, 28)  60000个矩阵组成的数组，每个矩阵由 28×28 个整数组成，每个这样的矩阵都是一张灰度图像，元素 取值范围为 0~255
print(train_images.shape)

# 数据类型
# uint8
print(train_images.dtype)

# 显示第四个数字
digit = train_images[40]
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()
