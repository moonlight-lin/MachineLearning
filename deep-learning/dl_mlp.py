import matplotlib.pyplot as plt

from tensorflow.keras import datasets
from tensorflow.keras import models
from tensorflow.keras import Sequential

from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten


# 导入样本数据
# 包、衣服、鞋等图片，60000 测试集，10000 训练集 (mnist 是 0~9 的数字图片，但有点简单，所以换成 fashion_mnist)
# 大小是 28*28
# 共 10 个分类
(train_x, train_y), (validation_x, validation_y) = datasets.fashion_mnist.load_data()

# 归一化
train_x = train_x / 255.0
validation_x = validation_x / 255.0

# 显示图片，可以看到是一个包
plt.figure()
plt.imshow(train_x[100])
plt.colorbar()
plt.grid(False)
plt.show()

# 创建模型
model = Sequential()
model.add(Flatten(input_shape=[28, 28]))    # 将图片拍平成一维数据
model.add(Dense(32, activation='relu'))     # 第一个神经层，32 个节点，激活函数是 ReLU
model.add(Dense(32, activation='relu'))     # 第二个神经层，32 个节点，激活函数是 ReLU
model.add(Dense(10, activation='softmax'))  # 输出层，10 个节点，因为有 10 个分类，激活函数是 softmax

# Dense 还有很多其他参数比如 kernel_regularizer，use_bias 等等

# 如果输入是一维数据，那么代码应该是
# model = Sequential()
# model.add(Dense(32, activation='relu', input_dim=784))     # 第一个神经层要指定输入的维度 28*28 = 784
# model.add(Dense(32, activation='relu'))
# model.add(Dense(10, activation='softmax'))

# 如果要加 sublayer 比如 dropout，可以这样
# from tensorflow.keras.layers import Dropout
# model.add(Dropout(0.5))

# 可以把模型画出来
# sudo apt-get install graphviz
# sudo pip install graphviz
# sudo pip install pydot_ng
# from tensorflow.keras.utils import plot_model
# plot_model(model)

# 创建模型时，也可以写成下面这种格式
# model = Sequential(
# [
#     Flatten(input_shape=[28, 28]),
#     Dense(32, activation='relu'),
#     Dense(32, activation='relu'),
#     Dense(10, activation='softmax')
# ])

# 编译模型，指定损失函数，优化器，也可以使用自己自定义的函数
# 这里的损失函数是交叉熵 sparse_categorical_crossentropy，也可以用 categorical_crossentropy
# 区别是前者不用 one-hot，后者用 one-hot，只是内部实现的区别
# metrics 表示如何衡量模型准确度，accuracy 表示真实值和预测值都是一个数字，还有其他类型
# 比如 categorical_accuracy 表示真实值是 one-hot 比如 [1, 0, 0] 而预测值是向量比如 [0.6, 0.3, 0.1]
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 训练
# 指定训练集和测试集
# 迭代 5 次 (计算完所有 60000 张图片算一次迭代)，每次计算最多用 100 张照片 (600次计算完成一个迭代)
model.fit(train_x, train_y, epochs=5, batch_size=100, validation_data=(validation_x, validation_y), verbose=1)

# 把模型有多少层，每层几个节点，多少参数，给打印出来
model.summary()

# 获取测试集的损失值，和准确度
loss, accuracy = model.evaluate(validation_x, validation_y, verbose=0)

# 预测
# 必须指定数组，如果只预测一个应该用 model.predict(validation_x[0:1])
result = model.predict(validation_x[0:100])

# 结果是 (100, 10) 表示有 100 个结果，每个值是一个 10 个值的列表表示属于不同分类 (共 10 个) 的概率
print(result.shape)

# 取第一个结果，可以看到属于最后一个分类的概率最大
# [1.54234713e-05 8.47201420e-08 7.35944195e-06 1.41576163e-06
#  3.19953301e-06 1.59782887e-01 2.61375098e-05 1.20886475e-01
#  4.77397395e-03 7.14502990e-01]
print(result[0])

# 保存模型
model.save('my_model')

# 读取模型
my_model = models.load_model('my_model')


