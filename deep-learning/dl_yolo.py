"""
用的 PyCharm

打开 PyCharm 的 Terminal 窗口，执行
   pip3.7.exe install -i https://pypi.mirrors.ustc.edu.cn/simple/ -r requirements.txt

如果有安装不了的比如
   opencv-python
   psutil

可以在 PyCharm 外安装好
   pip install -i https://pypi.mirrors.ustc.edu.cn/simple/ opencv-contrib-python  或  opencv-python
   pip install -i https://pypi.mirrors.ustc.edu.cn/simple/ psutil

然后把 cv2 和 psutil 目录拷过来，到项目的 venv\Lib\site-packages 下
每个包需要拷两个比如 psutil 和 psutil-5.9.4.dist-info

然后把 requirements.txt 里面的 opencv-python 和 psutil 注释掉，再 install
"""
import torch

# 下载模型
# 实际上不需要本地有 YOLO 的代码
# torch 会自动下载模型，存到当前目录
#     Downloading https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt to yolov5s.pt...
#     100%|██████████| 14.1M/14.1M [00:03<00:00, 4.42MB/s]
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# 打印模型结构
print(model)
print(type(model))

# 要识别的图片
#img = 'https://ultralytics.com/images/zidane.jpg'
img = 'C:\\Users\\Lin\\Desktop\\DeepLearning Training\\yolov5\\data\\images\\zidane.jpg'

# 识别
results = model(img)

# 输出识别的结果
# image 1/1: 720x1280 2 persons, 2 ties
# Speed: 16.0ms pre-process, 260.0ms inference, 2.0ms NMS per image at shape (1, 3, 384, 640)
results.print()

# 输出具体信息
"""
[         xmin        ymin         xmax        ymax  confidence  class    name
0  743.290405   48.343658  1141.756592  720.000000    0.879861      0  person
1  441.989624  437.336731   496.585083  710.036194    0.675119     27     tie
2  123.051147  193.238098   714.690735  719.771301    0.666693      0  person
3  978.989807  313.579468  1025.302856  415.526184    0.261517     27     tie]
"""
print(results.pandas().xyxy)

# 打开图片
# 关掉图片才会执行下一步
# results.show()

# Saved 1 image to runs\detect\exp4
results.save()
