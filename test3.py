import keras
import numpy as np
from keras.datasets import reuters
from keras import models
from keras import layers
import matplotlib.pyplot as plt

# 加载路途社数据
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)

print(train_data.shape)
print(test_data.shape)
print(train_data[10])

# 准备数据
def vectorize_sequences(sequences, dimension=10000):
    # Create an all-zero matrix of shape (len(sequences), dimension)
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.  # set specific indices of results[i] to 1s
    return results

# 将训练数据和测试数据向量化
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
print(x_train.shape)
print(x_train[0])
print(x_test.shape)

def to_one_hot(labels,dimension = 46):
    results = np.zeros((len(labels),dimension))
    for i,label in enumerate(labels):
        results[i,label] = 1.
    return results    

one_hot_train_labels = to_one_hot(train_labels)
one_hot_test_labels = to_one_hot(test_labels)

# 构建网络
model = models.Sequential()
model.add(layers.Dense(64,activation="relu",input_shape=(10000,)))
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(46,activation='softmax'))
# 编译模型
model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

# 验证
x_val = x_train[:1000]
print(x_val.shape)
partial_x_train = x_train[1000:]
print(partial_x_train.shape)

y_val = one_hot_train_labels[:1000]
print(y_val.shape)
partial_y_train = one_hot_train_labels[1000:]
print(partial_y_train.shape)

history = model.fit(partial_x_train,partial_y_train,epochs=20,batch_size=512,validation_data=(x_val,y_val))

# 绘制训练损失和验证损失
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1,len(loss) +1 )

plt.plot(epochs,loss,'bo',label ='Training loss')
plt.plot(epochs,val_loss,'b',label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend
plt.show()

