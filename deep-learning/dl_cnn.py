import matplotlib.pyplot as plt
import numpy as np

from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras import Sequential

from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPool2D

from tensorflow.keras.optimizers import Adam


# 读入训练集，和测试集
(train_x, train_y), (val_x, val_y) = keras.datasets.cifar100.load_data()

# <class 'numpy.ndarray'>
# <class 'numpy.ndarray'>
print(type(train_x))
print(type(train_y))

# (50000, 32, 32, 3)
# (50000, 1)
print(train_x.shape)
print(train_y.shape)

# (10000, 32, 32, 3)
# (10000, 1)
print(val_x.shape)
print(val_y.shape)

# 可以把图片画出来
plt.figure()
plt.imshow(train_x[500])
plt.colorbar()
plt.show()

# [[[255 255 255]
#   [255 255 255]
#   [255 255 255]
#   ...
#   [195 205 193]
#   [212 224 204]
#   [182 194 167]]
#
#  ...
#
#  [[ 87 122  41]
#   [ 88 122  39]
#   [101 134  56]
#   ...
#   [ 34  36  10]
#   [105 133  59]
#   [138 173  79]]]
print(train_x[0])

# 归一化
train_x = train_x.astype('float32') / 255
val_x = val_x.astype('float32') / 255

# [[[1.         1.         1.        ]
#   [1.         1.         1.        ]
#   [1.         1.         1.        ]
#   ...
#   [0.7647059  0.8039216  0.75686276]
#   [0.83137256 0.8784314  0.8       ]
#   [0.7137255  0.7607843  0.654902  ]]
#
#  ...
#
#  [[0.34117648 0.47843137 0.16078432]
#   [0.34509805 0.47843137 0.15294118]
#   [0.39607844 0.5254902  0.21960784]
#   ...
#   [0.13333334 0.14117648 0.03921569]
#   [0.4117647  0.52156866 0.23137255]
#   [0.5411765  0.6784314  0.30980393]]]
print(train_x[0])

# 构建模型
model = Sequential()

# 卷积层提供 Conv1D, Conv2D, Conv3D 三种
# 注意这里的 1D, 2D, 3D 不包含 channel
# 一张 RGB 图片是一个 2D 的面，有 3 个 channel，所以应该用 Conv2D
# 指定输入数据大小为 (32, 32, 3)
# 指定 64 个大小为 (3,3) 的卷积核 (可以只填一个 3 默认正方形)，每个核的 channel 自动取输入的 channel 也就是 3
# 使用 ReLU 作为激活函数
# 填充模式使用 same，这样输出的大小也是 32 * 32，输出的 channel 数则是 64 和卷积核个数一样
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
# 再做一层卷积 (可以连续做卷积)
# 卷积核 channel 数自动取 64
# 输入大小自动取上一层的输出大小，也就是 (32, 32, 64)
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
# 做归一化
model.add(BatchNormalization())
# 最大池化层，大小是 2*2 (也是默认值)，也可以只指定一个 2，两个方向的步长都是 2 (默认值是和 pool_size 一致)
# 输出大小是 (16, 16, 64)
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
# 可以加 Dropout
# model.add(Dropout(0.2))

# 再继续下一个卷积层
# 用 128 个卷积核，channel 数据自动取 64
# 输入自动取 (16, 16, 64)
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
# 再做一层卷积
# 卷积核 channel 数自动取 128
# 输入大小自动取上一层的输出大小，也就是 (16, 16, 128)
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
# 做归一化
model.add(BatchNormalization())
# 最大池化层，大小是 2*2 (也是默认值)，也可以只指定一个 2，两个方向的步长都是 2 (默认值是和 pool_size 一致)
# 输出大小是 (8, 8, 128)
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
# 可以加 Dropout
# model.add(Dropout(0.2))

# 拍平, 变成大小为 8*8*128 的一维数据
model.add(Flatten())

# 接全连接层
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.2))
# 继续全连接层
model.add(Dense(2048, activation='relu'))
model.add(Dropout(0.2))

# 输出层，因为是分类，激活函数用 softmax，共 100 个分类，所以节点是 100
model.add(Dense(100, activation='softmax'))

# 输出模型 (debug 用)
model.summary()

# 编译模型
# 因为是分类，损失函数用交叉熵
model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# 训练模型
# 在比较旧的机器，把第一个全连接层改成 2048 后，训练了 2.5 小时
# 训练集的准确率有 0.88 但测试集的准确率只有 0.45，模型还有待提高
model.fit(train_x, train_y, batch_size=500, epochs=100, validation_data=(val_x, val_y))

# 预测
index = 1001
print(train_y[index])
result = model.predict(train_x[index:index+1])  # 输入必须是列表，哪怕只有一个元素
print(result.shape)  # (1, 100)
max_type = np.where(result == np.max(result))   # 取最大值的下标
print(max_type)   # (array([0]), array([30])) 预测结果是分类 30

# 获取测试集的损失值，和准确度
loss, accuracy = model.evaluate(val_x, val_y, verbose=0)

# 保存模型
model_name = 'my_cnn_model_' + str(loss) + '_' + str(accuracy)
model.save(model_name)

# 读入模型
my_model = models.load_model(model_name)
