# 导入相关库
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import os
import cv2
import paddle
import numpy as np
from PIL import Image
import paddle.nn as nn
import matplotlib.pyplot as plt
import paddle.vision.transforms as T
from paddle.vision.datasets import DatasetFolder,ImageFolder
import paddle.optimizer as optim
import pandas as pd

# 数据加载部分
# 这部分代码取决于你的数据格式和存储方式
# 假设你已经有了一个数据加载器，它能够加载并预处理你的图片数据
def no_axis_show(img, title='', cmap=None):
  # imshow, 縮放模式為nearest。
  fig = plt.imshow(img, interpolation='nearest', cmap=cmap)
  # 不要显示axis
  fig.axes.get_xaxis().set_visible(False)
  fig.axes.get_yaxis().set_visible(False)
  plt.title(title)


#标签映射
titles = ['horse', 'bed', 'clock', 'apple', 'cat', 'television', 'dog', 'dolphin', 'spider']


# 训练集预处理
def source_transform(imge):
    # 转灰色: Canny 不吃 RGB。
    img = T.to_grayscale(imge)
    # cv2 不吃 skimage.Image，因此转成np.array后再做cv2.Canny
    img = cv2.Canny(np.array(img), 170, 300)
    # 重新np.array 转回 skimage.Image
    img = Image.fromarray(np.array(img))
    # 随机水平翻转 (Augmentation)
    RHF= T.RandomHorizontalFlip(0.5)
    img = RHF(img)
    # 旋转15度内 (Augmentation)，旋转后空的地方补0
    RR = T.RandomRotation(15, fill=(0,))
    img = RR(img)
    # 最后Tensor供model使用。
    tensor = T.ToTensor()

    return tensor(img)

# 测试集预处理
target_transform = T.Compose([
    # 转灰阶:
   T.Grayscale(),
    # 缩放: 因为source data是32x32，我们把target data的28x28放大成32x32。
    T.Resize((32, 32)),
    # 随机水平翻转(Augmentation)
    T.RandomHorizontalFlip(0.5),
    # 旋转15度内 (Augmentation)，旋转后空的地方补0
    T.RandomRotation(15, fill=(0,)),
    # 最后Tensor供model使用。
    T.ToTensor(),
])


# 生成数据集
source_dataset = DatasetFolder('./train_data/', transform=source_transform) # DatasetFolder 用于读取训练集，读取的时候图片和标签
target_dataset = DatasetFolder('./testdata_raw/', transform=target_transform) # ImageFolder 用于读取测试集，读取的时候只有图片

# 数据加载器定义
source_dataloader = paddle.io.DataLoader(source_dataset, batch_size=50, shuffle=True)
target_dataloader = paddle.io.DataLoader(target_dataset, batch_size=50, shuffle=True)
test_dataloader = paddle.io.DataLoader(target_dataset, batch_size=100, shuffle=False)



# matplotlib inline
# 展示生成并经过预处理的的source_dataset和source_loader
print('=============source_dataset=============')
#由于使用了DatasetFolder，训练集这里有图片和标签两个参数image,label
for image, label in source_dataset:
    print('image shape: {}, label: {}'.format(image.shape,label))
    print('训练集数量:',len(source_dataset))
    print('图片：',image)
    print('标签：',label)
    plt.imshow(image.numpy().squeeze(),cmap='gray')
    break


#source_loader的信息
print('=============source_dataloader=============')
for batch_id, (data,label) in enumerate(source_dataloader):
    print('一个batch的图片：',data.shape)    # 索引[0]存放图片
    print('一个batch的标签个数：',label.shape)   #索引[1]存放标签
    print('图片：',data[0].shape)
    break

# no_axis_show(x_data.numpy().squeeze(),title='process image', cmap='gray')


# 展示生成并经过预处理的target_dataset和target_dataloader
print('=============target_dataset=============')

for image_,_ in target_dataset:
    print('image shape: {}'.format(image_.shape))
    print('测试集数量:',len(target_dataset))
    plt.imshow(image_.numpy().squeeze(),cmap='gray')
    print('图片：',image_)
    break


#target_dataloader的信息
print('=============target_dataloader=============')
for batch_id, (data_1,label_1) in enumerate(target_dataloader):
    # print('一个batch的图片：',data[0].shape)
    print('一个batch的图片：',data_1.shape)
    print('一张图片的形状：',data_1[0].shape)
    print(label_1)

    break




# 模型定义部分
class FeatureExtractor(nn.Layer): #特征提取器
  '''
  从图片中抽取特征
  input [batch_size ,1,32,32]
  output [batch_size ,512]
  '''

  def __init__(self):
    super(FeatureExtractor, self).__init__()

    self.conv = nn.Sequential(
      nn.Conv2D(in_channels=1, out_channels=64, kernel_size=3, padding=1, stride=1),
      # [batch_size ,64,32,32] (32-3+2*1)/1 + 1
      nn.BatchNorm2D(64),
      nn.ReLU(),
      nn.MaxPool2D(kernel_size=2),  # [batch_size ,64,16,16]

      nn.Conv2D(64, 128, 3, 1, 1),  # [batch_size ,128,16,16]
      nn.BatchNorm2D(128),
      nn.ReLU(),
      nn.MaxPool2D(2),  # [batch_size ,128,8,8]

      nn.Conv2D(128, 256, 3, 1, 1),  # [batch_size ,256,8,8]
      nn.BatchNorm2D(256),
      nn.ReLU(),
      nn.MaxPool2D(2),  # [batch_size ,256,4,4]

      nn.Conv2D(256, 256, 3, 1, 1),  # [batch_size ,256,4,4]
      nn.BatchNorm2D(256),
      nn.ReLU(),
      nn.MaxPool2D(2),  # [batch_size ,256,2,2]

      nn.Conv2D(256, 512, 3, 1, 1),  # [batch_size ,512,2,2]
      nn.BatchNorm2D(512),
      nn.ReLU(),
      nn.MaxPool2D(2),  # [batch_size ,512,1,1]
      nn.Flatten()  # [batch_size ,512]
    )

  def forward(self, x):
    x = self.conv(x)  # [batch_size ,256]
    return x


class LabelPredictor(nn.Layer): #标签预测器
  '''
  预测图像是什么动物
  '''

  def __init__(self):
    super(LabelPredictor, self).__init__()

    self.layer = nn.Sequential(
      nn.Linear(512, 512),
      nn.ReLU(),

      nn.Linear(512, 512),
      nn.ReLU(),

      nn.Linear(512, 10),
    )

  def forward(self, h):
    c = self.layer(h)
    return c


class DomainClassifier(nn.Layer): #Domain Classifier领域的分类器
  '''预测时手绘还是真实图片'''

  def __init__(self):
    super(DomainClassifier, self).__init__()

    self.layer = nn.Sequential(
      nn.Linear(512, 512),
      nn.BatchNorm1D(512),
      nn.ReLU(),

      nn.Linear(512, 512),
      nn.BatchNorm1D(512),
      nn.ReLU(),

      nn.Linear(512, 512),
      nn.BatchNorm1D(512),
      nn.ReLU(),

      nn.Linear(512, 512),
      nn.BatchNorm1D(512),
      nn.ReLU(),

      nn.Linear(512, 1),
    )

  def forward(self, h):
    y = self.layer(h)
    return y


# 模型配置
# 模型实例化
feature_extractor = FeatureExtractor()
label_predictor = LabelPredictor()
domain_classifier = DomainClassifier()
class_criterion = nn.CrossEntropyLoss()
domain_criterion = nn.BCEWithLogitsLoss()
# 定义优化器
optimizer_F = optim.Adam(learning_rate=0.0001, parameters=feature_extractor.parameters())
optimizer_C = optim.Adam(learning_rate=0.0001, parameters=label_predictor.parameters())
optimizer_D = optim.Adam(learning_rate=0.0001, parameters=domain_classifier.parameters())


# 加载测试集
# 已有 test_dataloader = paddle.io.DataLoader(target_dataset, batch_size=100, shuffle=False)
# 加载模型
# 已有feature_extractor = FeatureExtractor()
# 已有label_predictor = LabelPredictor()
feature_extractor.set_state_dict(paddle.load("model0/feature_extractor_final.pdparams"))
label_predictor.set_state_dict(paddle.load("model0/label_predictor_final.pdparams"))

# 训练部分
# 这部分代码取决于你的训练策略
# 假设你已经训练了你的模型，并且保存在了feature_extractor变量中
# feature_extractor = ...

# 可视化部分
# 可视化部分
def visualize_feature_distribution(real_images_loader, hand_drawn_images_loader, feature_extractor):
    # 使用特征提取器获取图片的特征
    for real_images, _ in real_images_loader:
        real_features = feature_extractor(real_images)
        break

    for hand_drawn_images, _ in hand_drawn_images_loader:
        hand_drawn_features = feature_extractor(hand_drawn_images)
        break

    # 将特征转换为numpy数组
    real_features = real_features.detach().numpy()
    hand_drawn_features = hand_drawn_features.detach().numpy()

    # 使用t-SNE降维
    tsne = TSNE(n_components=2, random_state=0)
    real_2d = tsne.fit_transform(real_features)
    hand_drawn_2d = tsne.fit_transform(hand_drawn_features)

    # 创建一个散点图
    plt.figure(figsize=(6, 6))
    plt.scatter(real_2d[:, 0], real_2d[:, 1], color='r', label='Real Images')
    plt.scatter(hand_drawn_2d[:, 0], hand_drawn_2d[:, 1], color='b', label='Hand Drawn Images')
    plt.legend()
    plt.show()

real_images_loader = source_dataloader
hand_drawn_images_loader = target_dataloader
# 使用你的数据和模型调用可视化函数
visualize_feature_distribution(real_images_loader, hand_drawn_images_loader, feature_extractor)