import keras

keras.__version__
from keras.datasets import mnist
from keras import models
from keras import layers
from keras import optimizers
from keras.utils import to_categorical

# 输入图像保存在 float32 格式的 Numpy 张量中，形状分别为 (60000, 784)(训练数据)和 (10000, 784)(测试数据)
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

print(train_labels.shape)
print(train_labels[0])
# 将向量转化为矩阵
train_labels = to_categorical(train_labels)
print(train_labels.shape)
print(train_labels[0])
test_labels = to_categorical(test_labels)

# 构建网络
# 这个网络包含两个 Dense 层，每层都对输入数据进行一些简单的张量运算， 这些运算都包含权重张量。权重张量是该层的属性，里面保存了网络所学到的知识(knowledge)
model = models.Sequential()
# 现在模型就会以shape为(*,28*28)的数组作为输入
# 其输出的数组shape为（*，32）
model.add(layers.Dense(32, activation='relu', input_shape=(28 * 28,)))
# 10 路 softmax 层，它将返回一个由 10 个概率值(总和为 1)组成的数组。 每个概率值表示当前数字图像属于 10 个数字类别中某一个的概率
model.add(layers.Dense(10, activation='softmax'))

# 网络的编译
# 现在你明白了，mse 是损失函数，是用于学习权重张量的反馈 信号，在训练阶段应使它最小化。
# 你还知道，减小损失是通过小批量随机梯度下降来实现的。 梯度下降的具体方法由第一个参数给定，即 rmsprop 优化器
model.compile(optimizer=optimizers.RMSprop(lr=0.001), loss='mse', metrics=['accuracy'])
# 下面是训练循环
# 现在你明白在调用 fit 时发生了什么:网络开始在训练数据上进行迭代(每个小批量包含 128 个样本)，共迭代 5 次[在所有训练数据上迭代一次叫作一个轮次(epoch)]。
# 在每次迭代过程中，网络会计算批量损失相对于权重的梯度，并相应地更新权重。5 轮之后，网络进行了 2345 次梯度更新(每轮 469 (60000个样本，60000/128=468.75) 次)，网络损失值将变得足够小，
# 使得网络能够以很高的精度对手写数字进行分类。
# 模型以数组shape为(128,28*28)为输入。第一层输出数组为（128，32）
# x矩阵为 128*784，weight矩阵为784*32，积为128*32
model.fit(train_images, train_labels, epochs=5, batch_size=128)

test_loss, test_acc = model.evaluate(test_images, test_labels)
