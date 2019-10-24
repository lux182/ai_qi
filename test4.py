from keras.datasets import boston_housing
from keras import models
from keras import layers
import numpy as np
import matplotlib.pyplot as plt
(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

print(train_data.shape)
print(test_data.shape)
print(train_targets.shape)

# 将取值范围差异很大的数据进行特征标准化
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std
# 用于测试数据的均值和标准差都是在训练数据上计算得到的。在工作流中，你不能使用在测试数据上计算得到的任何结果
test_data -= mean
test_data /= std

# 定义模型


def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu',
                           input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    # 最后一层只有一个单元，没有激活，是一个线性层。这是标量回归的典型配置
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model


# K折验证
k = 4
num_val_samples = len(train_data) // k

# num_epochs = 100
# all_scores = []

# for i in range(k):
#     print('processing fold #', i)
#     val_data = train_data[i * num_val_samples:(i+1) * num_val_samples]
#     val_targets = train_targets[i * num_val_samples:(i + 1) * num_val_samples]

#     partial_train_data = np.concatenate(
#         [train_data[:i * num_val_samples],
#          train_data[(i+1) * num_val_samples:]],
#         axis=0
#     )

#     partial_train_targets = np.concatenate(
#         [train_targets[:i * num_val_samples],
#          train_targets[(i+1) * num_val_samples:]],
#         axis=0
#     )

#     model = build_model()
#     model.fit(partial_train_data, partial_train_targets,
#               epochs=num_epochs, batch_size=1, verbose=0)
#     val_mse,val_mae = model.evaluate(val_data,val_targets,verbose=0)
#     all_scores.append(val_mae)

# print(all_scores)
# print(np.mean(all_scores))

num_epochs = 500
all_mae_histories = []

for i in range(k):
    print('processing fold #', i)
    val_data = train_data[i * num_val_samples:(i+1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples:(i + 1) * num_val_samples]

    partial_train_data = np.concatenate(
        [train_data[:i * num_val_samples],
         train_data[(i+1) * num_val_samples:]],
        axis=0
    )

    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples],
         train_targets[(i+1) * num_val_samples:]],
        axis=0
    )

    model = build_model()
    history = model.fit(partial_train_data, partial_train_targets, validation_data=(val_data, val_targets),
                        epochs=num_epochs, batch_size=1, verbose=0)
    mae_history = history.history['val_mean_absolute_error']
    all_mae_histories.append(mae_history)

average_mae_history = [np.mean([x[i] for x in all_mae_histories])
                       for i in range(num_epochs)]


plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()


def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1-factor))
        else:
            smoothed_points.append(point)
    return smoothed_points


smooth_mae_history = smooth_curve(average_mae_history[10:])
plt.plot(range(1, len(smooth_mae_history)+1), smooth_mae_history)
plt.xlabel("Epochs")
plt.ylabel("Validation MAE")
plt.show()
