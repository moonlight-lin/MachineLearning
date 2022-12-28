from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.vgg16 import VGG16


# 稠密网络
#
# include_top：
#    True：自动加 avg pool，自动加全连接层，自动使用 softmax，并且输入为 (224, 224, 3)
#    False：自己设计全连接层，自己定义 input_shape，pool 类型，激活函数，分类总数
#    默认是 True
# weights：
#    imagenet：代表使用系统已经训练好的 weight，且如果 include_top 为 True，那么分类总数就是 1000
#    path：使用自己训练好的 weight，path 是这个 weight 的存储路径
#    None：随机初始化，模型建好后需要进行训练
model = Sequential()
model.add(DenseNet121(include_top=False, pooling='avg', weights=None, input_shape=(224, 224, 3)))
model.add(Dense(10, activation='softmax'))

# 指定分类总数
# 不使用训练好的 weight
# 自动加 avg pool，自动加全连接层，自动使用 softmax，并且输入为 (224, 224, 3)
model = DenseNet121(weights=None, classes=10)

# 直接使用系统训练好的模型
model = DenseNet121(weights='imagenet')


# 残差网络
model = Sequential()
model.add(ResNet50(include_top=False, pooling='avg', weights=None, input_shape=(224, 224, 3)))
model.add(Dense(10, activation='softmax'))

model = ResNet50(weights=None, classes=10)
model = ResNet50(weights='imagenet')


# inception v3 网络
model = Sequential()
model.add(InceptionV3(include_top=False, pooling='avg', weights=None, input_shape=(299, 299, 3)))
model.add(Dense(10, activation='softmax'))

model = InceptionV3(weights=None, classes=10)
model = InceptionV3(weights='imagenet')


# VGG 网络
model = Sequential()
model.add(VGG16(include_top=False, pooling='avg', weights=None, input_shape=(299, 299, 3)))
model.add(Dense(10, activation='softmax'))

model = VGG16(weights=None, classes=10)
model = VGG16(weights='imagenet')
