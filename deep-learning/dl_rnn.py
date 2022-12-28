import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout

from tensorflow.keras.optimizers import Adam


# 打印 pd 数据的时候，如果列太多，会隐藏部分列，设置成不隐藏
pd.set_option('expand_frame_repr', False)

# 读数据
# 假设数据列分别是 日期、温度、湿度、气压、风速、风向、降水量、云量、空气质量
# 假设空气质量不是独立的，而是和前面的天气情况有关
# 然后希望能用前 10 天每天的 (温度、湿度、气压、风速、风向、降水量、云量、空气质量) 预测第 11 天的空气质量
# index_col=[0] 将第一列也就是 date 设置成 index
# 通过 parse_dates 转换日期
data = pd.read_csv('air_condition.csv', index_col=[0], parse_dates=True)
data.head()

'''
如果没指定 index 也没指定 parse_dates 那数据如下，会自动添加行号作为 index

             date  temperature  humidity  pressure  wind_speed wind_direction  rain  cloud  air_condition
0  2020/1/1 12:00           20        60      1000         2.1              W     0     50             50
1  2020/1/1 13:00           21        50      1010         1.4              E     0     60             60
2  2020/1/1 14:00           22        55      1020         1.0             SE    10     90             85
3  2020/1/1 15:00           20        65      1030         2.5              S     5     70             77
4  2020/1/1 16:00           21        60      1010         2.0              W     0     50             65

如果指定 index 和 parse_dates 那数据如下，使用 date 做 index，并转换日期

                     temperature  humidity  pressure  wind_speed wind_direction  rain  cloud  air_condition
date                                                                                                       
2020-01-01 12:00:00           20        60      1000         2.1              W     0     50             50
2020-01-01 13:00:00           21        50      1010         1.4              E     0     60             60
2020-01-01 14:00:00           22        55      1020         1.0             SE    10     90             85
2020-01-01 15:00:00           20        65      1030         2.5              S     5     70             77
2020-01-01 16:00:00           21        60      1010         2.0              W     0     50             65
'''

# 将风速转换成数值
encoder = LabelEncoder()
data["wind_direction"] = encoder.fit_transform(data["wind_direction"])
data.head()
'''
                     temperature  humidity  pressure  wind_speed  wind_direction  rain  cloud  air_condition
date                                                                                                        
2020-01-01 12:00:00           20        60      1000         2.1               3     0     50             50
2020-01-01 13:00:00           21        50      1010         1.4               0     0     60             60
2020-01-01 14:00:00           22        55      1020         1.0               2    10     90             85
2020-01-01 15:00:00           20        65      1030         2.5               1     5     70             77
2020-01-01 16:00:00           21        60      1010         2.0               3     0     50             65
'''

'''
# 也可以用 one-hot 转

data = pd.get_dummies(data)
data.head()

                     temperature  humidity  pressure  wind_speed  rain  cloud  air_condition  wind_direction_E  wind_direction_S  wind_direction_SE  wind_direction_W
date                                                                                                                                                                 
2020-01-01 12:00:00           20        60      1000         2.1     0     50             50                 0                 0                  0                 1
2020-01-01 13:00:00           21        50      1010         1.4     0     60             60                 1                 0                  0                 0
2020-01-01 14:00:00           22        55      1020         1.0    10     90             85                 0                 0                  1                 0
2020-01-01 15:00:00           20        65      1030         2.5     5     70             77                 0                 1                  0                 0
2020-01-01 16:00:00           21        60      1010         2.0     0     50             65                 0                 0                  0                 1
'''

'''
# 或者通过 values 处理
values = data.values
values[:, 4] = encoder.fit_transform(values[:, 4])

array([[20, 60, 1000, 2.1, 3, 0, 50, 50],
       [21, 50, 1010, 1.4, 0, 0, 60, 60],
       [22, 55, 1020, 1.0, 2, 10, 90, 85],
       [20, 65, 1030, 2.5, 1, 5, 70, 77],
       [21, 60, 1010, 2.0, 3, 0, 50, 65]], dtype=object)
       
data = pd.DataFrame(values, index=data.index, columns=data.columns)
'''

# 再做归一化
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)
data = pd.DataFrame(data_scaled, index=data.index, columns=data.columns)
data.head()

'''
# 也可以这样
scaler = MinMaxScaler().fit(data.values)
data_scaled = scaler.transform(data.values)
data = pd.DataFrame(data_scaled, index=data.index, columns=data.columns)
data.head()
'''

'''
                     temperature  humidity  pressure  wind_speed  wind_direction  rain  cloud  air_condition
date                                                                                                        
2020-01-01 12:00:00          0.0  0.666667  0.000000    0.733333        1.000000   0.0   0.00       0.000000
2020-01-01 13:00:00          0.5  0.000000  0.333333    0.266667        0.000000   0.0   0.25       0.285714
2020-01-01 14:00:00          1.0  0.333333  0.666667    0.000000        0.666667   1.0   1.00       1.000000
2020-01-01 15:00:00          0.0  1.000000  1.000000    1.000000        0.333333   0.5   0.50       0.771429
2020-01-01 16:00:00          0.5  0.666667  0.333333    0.666667        1.000000   0.0   0.00       0.428571
'''

##########
# pandas 还可以做很多其他操作，比如去掉有异常值的数据，去掉有丢失值的数据，对丢失值做插值补全等等
##########

##########
# 下面取每 10 天的所有特征数据作为一个样本，第 11 天的 air_condition 作为标签
# 假设共 n 天数据，那么 x 的大小是 (n-10, 10, 8), 而 y 的大小是 (n-10, 1)
##########
x, y = list(), list()
values = data.values
value_length = len(values)

for i in range(value_length):

    # 每个样本需要 10 天的 x 和 1 天的 y 共 11 天数据
    if i + 10 > value_length - 1:
        break

    x_slice = [ix for ix in range(i, i + 10)]

    # 获取每个样本的输入序列 seq_x 和标签 seq_y
    seq_x = values[x_slice, :]
    seq_y = values[i + 10, 7]

    x.append(seq_x)
    y.append(seq_y)

x = np.array(x)
y = np.array(y)

print(x.shape)
print(y.shape)
print(type(x))
print(type(y))
'''
(1000, 10, 8)
(1000,)
<class 'numpy.ndarray'>
<class 'numpy.ndarray'>
'''

# 下面构建 LSTM 模型
model = Sequential()

# units 代表隐藏层的神经元数量，也是隐藏层的输出维度
# input_shape 代表输入的维度，这里是连续 10 天，每天 8 个指标，所以是 (10, 8)
# activation 代表激活函数，默认是 tanh，可以通过 activation='relu' 改变激活函数
# return_sequences 为 true 表示每输入一天的数据，都要有对应的输出
#                  为 false 表示要输入 10 天的数据，才有对应的输出
#                  由于这里设置多层 LSTM，所以第一层设置为 true，要求每天的输入都有对应输出
#                  如果只有一层，或者是最后一层，就设置为 false (默认就是 false)，表示输入 10 天数据，才有对应的输出
model.add(LSTM(units=20, return_sequences=True, input_shape=(10, 8)))
# 可以在两层 LSTM 之间加入 Dropout 层
model.add(Dropout(rate=0.1))
# 第二层不用指定 input，会自动按照第一层的输出设置，这里应该是 (10, 20)，
# 10 是因为第一层每天都有对应输出，20 是因为第一层 units 是 20
model.add(LSTM(units=10))
# 和全连接层间也可以加入 Dropout 层
model.add(Dropout(rate=0.2))
# 后面接全连接层，不需要指定输入维度，自动按上层的输出设置，这里是 (1, 10)，
# 因为第二层的 units 是 10，并且 10 天输入数据只对应一个输出
model.add(Dense(units=5, activation='relu'))
# 最后是一维输出，只预测第 11 天的空气质量一个值，如果是分类的输出，这里的 units 就是分类数，激活函数就是 softmax
model.add(Dense(units=1, activation='relu'))

# 编译模型
model.compile(optimizer=Adam(learning_rate=0.005), loss='mae', metrics=['accuracy'])
model.summary()

# 训练模型
model.fit(x, y, epochs=100, verbose=1, shuffle=False)

# 预测，注意这里每个数据，比如 x[5] 都是一个 (10, 8) 维的数据，每个返回是一个值，这里 predict 是 (5,1) 维数据
predict = model.predict(x[5:10], verbose=1)

print(type(predict))
'''
<class 'numpy.ndarray'>
'''



'''
可以把样本数据拆成多个数据集

KFold
-----
比如把数据划分成 4 分，其中一个做验证集，其他3个合成做训练集，然后换一个做验证集，剩下3个做训练集，如此可以训练出 4 个模型，
预测的时候，如果预测值，就取4个模型的平均，如果分类，就按多数取胜

TimeSeriesSplit
----
用于 RNN，类似 KFold 但要考虑顺序，比如分成 4 分，
用第一分做训练，第二份做验证，然后用第一份和第二份合成做训练，用第3份做验证，再用 1~3 份做训练集，用第 4 份做验证集，
这样可以训练出 3 个模型
'''

timeSeriesSplit = TimeSeriesSplit(n_splits=4)

model_list = []
predict_list = []
for train, test in timeSeriesSplit.split(x, y):
    # 构建模型
    model = Sequential()
    model.add(LSTM(units=20, input_shape=(10, 8)))
    model.add(Dense(units=1, activation='relu'))
    model.compile(optimizer=Adam(learning_rate=0.005), loss='mae', metrics=['accuracy'])

    # 用训练集训练
    model.fit(x[train], y[train], epochs=100, verbose=1, shuffle=False)

    # 用测试集验证
    predict = model.predict(x[test], verbose=1)

    # 保存模型和验证结果
    model_list.append(model)
    predict_list.append(predict)

# 预测
predict = None
for model in model_list:
    predict_y = model.predict(x[5:10], verbose=1)
    if predict is None:
        predict = predict_y
    else:
        predict = predict + predict_y

predict = predict/len(model_list)

