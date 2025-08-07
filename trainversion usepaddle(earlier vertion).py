# 导入相关库
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


def no_axis_show(img, title='', cmap=None):
  # imshow, 縮放模式為nearest。
  fig = plt.imshow(img, interpolation='nearest', cmap=cmap)
  # 不要显示axis
  fig.axes.get_xaxis().set_visible(False)
  fig.axes.get_yaxis().set_visible(False)
  plt.title(title)


#标签映射
titles = ['horse', 'bed', 'clock', 'apple', 'cat', 'television', 'dog', 'dolphin', 'spider']
plt.figure(figsize=(18, 18))
for i in range(9):
  plt.subplot(1, 9, i+1)
  fig = no_axis_show(plt.imread(f'./train_data/{i}/{500*i}.bmp'), title=titles[i])

plt.figure(figsize=(18, 18))
for i in range(9):
    plt.subplot(1, 9, i + 1)
    fig = no_axis_show(plt.imread(f'./testdata_raw/0/0000{i}.bmp'), title='none')

plt.show()



plt.figure(figsize=(18, 18))

original_img = plt.imread(f'./train_data/0/197.bmp')
plt.subplot(1, 5, 1)
no_axis_show(original_img, title='original')

gray_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
plt.subplot(1, 5, 2)
no_axis_show(gray_img, title='gray scale', cmap='gray')


canny_50100 = cv2.Canny(gray_img, 50, 100)
plt.subplot(1, 5, 3)
no_axis_show(canny_50100, title='Canny(50, 100)', cmap='gray')

canny_150200 = cv2.Canny(gray_img, 150, 200)
plt.subplot(1, 5, 4)
no_axis_show(canny_150200, title='Canny(150, 200)', cmap='gray')

canny_250300 = cv2.Canny(gray_img, 250, 300)
plt.subplot(1, 5, 5)
no_axis_show(canny_250300, title='Canny(250, 300)', cmap='gray')

plt.show()



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

#调用一下数据预处理函数
original_img = Image.open(f'./train_data/0/197.bmp')
print('原来的照片形状：',np.array(original_img).shape)

process = source_transform(original_img)
print('预处理后的照片形状：',process .shape)
print(process)

plt.subplot(1,2,1)
no_axis_show(process .numpy().squeeze(), title='process image',cmap='gray')

plt.subplot(1,2,2)
no_axis_show(original_img, title='origimal image', cmap='gray')

plt.show()



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



# 定义训练函数
import paddle
def train_epoch(source_dataloader, target_dataloader, lamb):
    '''
      Args:
        source_dataloader: source data的dataloader
        target_dataloader: target data的dataloader
        lamb: 调控adversarial的loss系数。
    '''
    running_D_loss, running_F_loss = 0.0, 0.0
    total_hit, total_num = 0.0, 0.0

    for i, ((source_data, source_label), (target_data,_)) in enumerate(zip(source_dataloader, target_dataloader)):
        mixed_data = paddle.concat([source_data, target_data], axis=0)
        domain_label = paddle.zeros([source_data.shape[0] + target_data.shape[0], 1]).cpu() #原来是cuda，试着改成cpu，用cpu硬刚，不过colab上有很香的gpu
        # 设定source data的label为1
        domain_label[:source_data.shape[0]] = 1

        # Step 1 : 训练Domain Classifier
        feature = feature_extractor(mixed_data)
        # 因为我们在Step 1不需要训练Feature Extractor，所以把feature detach
        #这样可以把特征抽取过程的函数从当前计算图分离，避免loss backprop传递过去。
        domain_logits = domain_classifier(feature.detach())
        loss = domain_criterion(domain_logits, domain_label)
        running_D_loss += float(loss)
        loss.backward()
        optimizer_D.step()

        # Step 2 : 训练Feature Extractor和Domain Classifier
        class_logits = label_predictor(feature[:source_data.shape[0]])
        domain_logits = domain_classifier(feature)
        # loss为原本的class CE - lamb * domain BCE，相減的原因是我们希望特征能够使得domain_classifier分不出来输入的图片属于哪个领域
        loss = class_criterion(class_logits, source_label) - lamb * domain_criterion(domain_logits, domain_label)
        running_F_loss += float(loss)
        loss.backward()
        optimizer_F.step()
        optimizer_C.step()
        #训练了一轮，清空所有梯度信息
        optimizer_D.clear_grad()
        optimizer_F.clear_grad()
        optimizer_C.clear_grad()
        # return class_logits,source_label  #测试
        bool_eq = paddle.argmax(class_logits, axis=1) == source_label.squeeze()
        total_hit += bool_eq.sum().item()
        total_num += source_data.shape[0]
        print(i, end='\r')

    return running_D_loss / (i+1), running_F_loss / (i+1), total_hit / total_num


# 训练250 epochs
train_D_loss_history, train_F_loss_history, train_acc_history = [], [], []
for epoch in range(250):
  train_D_loss, train_F_loss, train_acc = train_epoch(source_dataloader, target_dataloader, lamb=0.1)

  train_D_loss_history.append(train_D_loss)
  train_F_loss_history.append(train_F_loss)
  train_acc_history.append(train_acc)

  epoch = epoch + 1
  if epoch % 50 == 0:
    paddle.save(feature_extractor.state_dict(), "ckp/{}ckp_feature_extractor.pdparams".format(str(epoch)))
    paddle.save(label_predictor.state_dict(), "ckp/{}ckp_label_predictor.pdparams".format(str(epoch)))

  print('epoch {:>3d}: train D loss: {:6.4f}, train F loss: {:6.4f}, acc {:6.4f}'.format(epoch, train_D_loss,
                                                                                         train_F_loss, train_acc))



#保存模型
paddle.save(feature_extractor.state_dict(), "model0/feature_extractor_final.pdparams")
paddle.save(label_predictor.state_dict(), "model0/label_predictor_final.pdparams")



#分开绘制三条曲线
epochs = range(epoch)
# 模型训练可视化
def draw_process(title,color,iters,data,label):
    plt.title(title, fontsize=20)  # 标题
    plt.xlabel("epochs", fontsize=15)  # x轴
    plt.ylabel(label, fontsize=15)  # y轴
    plt.plot(iters, data,color=color,label=label)   # 画图
    plt.legend()
    plt.grid()
    plt.savefig('{}.jpg'.format(title))
    plt.show()

# Domain Classifier train loss
draw_process("train D loss","green",epochs,train_D_loss_history,"loss")
# Feature Extrator train loss
draw_process("train F loss","green",epochs,train_F_loss_history,"loss")
# Label Predictor的train accuracy
draw_process("train acc","red",epochs,train_acc_history,"accuracy")

plt.show()