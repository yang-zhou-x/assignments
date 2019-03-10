# -*- coding: utf-8 -*-
"""
Created on 2018-12
@author: Yang Zhou
"""

import os
import time
import random
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch as t
import torch.nn.functional as f
from torch import nn
from torch import optim
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms, utils
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

'''
1.简介
'''
# 本文通过建立残差网络模型ResNet34，对114人的头部图片进行分类预测，即输入测试图片后返回人物的对应标签。
# 数据来源：CVL Face Database by Peter Peer，University of Ljubljana。

'''
2.实验环境
'''
# 本文基于python3.7，使用pytorch框架建立残差网络模型ResNet34，并利用NVIDIA CUDA进行训练。
# CUDA版本为9.2，cuDNN版本为7.3.1。代码兼容CPU和GPU。GPU所在平台为Windows10。
device = t.device('cuda:0' if t.cuda.is_available() else 'cpu')
print(device)

'''
3.数据准备
'''
# 数据集包含114人、每人7张（极少数为6张或7张以上）照片。114人的标签为0-113。
# 图片的存放形式为每个人的照片存放在1个文件夹中，共114个文件夹。

# 初始路径
path_init = '~/'


# 3.1.划分训练集和测试集
# 3.1.1.删除无关文件
# 移除每个文件夹中无关的HTM格式的文件
def rm_htm(path):
    """
    删除单个文件夹中的HTM格式文件。
    :param path: 图片所在文件夹的路径
    """
    for name in os.listdir(path):
        if name.endswith(('.HTM', '.htm')):
            os.remove(os.path.join(path, name))


def rm_all(path):
    """
    循环删除所有114个文件夹中的HTM格式的文件
    :param path: 初始路径
    """
    for i in range(1, 115, 1):
        path_used = os.path.join(path, 'database', str(i))
        rm_htm(path_used)


# 删除所有文件夹中HTM格式的文件
rm_all(path=path_init)


# 3.1.2.建立测试集
# 从114个文件夹中，分别随机抽取30%（即2张）的图片，移动到新的114个文件夹中，作为测试集数据。剩余的70%作为训练集。
def mv_photo(from_path, to_path):
    """
    移动照片到新文件夹
    :param from_path: 图片所在文件夹的路径
    :param to_path: 新文件夹所在的路径
    """
    names = os.listdir(from_path)
    samples = random.sample(names, 2)  # 2张
    os.mkdir(to_path)
    for name in samples:
        shutil.move(os.path.join(from_path, name), to_path)


def mv_all(path):
    """
    移动所有的测试图片到新文件夹
    :param path: 初始路径
    """
    os.mkdir(os.path.join(path, 'test'))  # 创建测试集文件夹
    for i in range(1, 115, 1):
        from_path = os.path.join(path, 'database', str(i))
        to_path = os.path.join(path, 'test', str(i))
        mv_photo(from_path=from_path, to_path=to_path)


# 移动测试图片到指定位置
mv_all(path=path_init)

# 3.2.读取图片数据
# 根据资料，图片像素统一为640*480，不需要额外处理图片大小。
# 训练集
train_set = ImageFolder(root=os.path.join(path_init, 'database'),
                        transform=transforms.Compose([
                            transforms.CenterCrop(size=(300, 300)),
                            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                            transforms.ToTensor()
                        ]))
# 查看训练集图片总数：
print(len(train_set))
# 查看单张照片数据的形状：
print(train_set[0][0].size())  # RGB三通道照片
# 查看该照片对应的标签，即人物编号：
print(train_set[0][1])  # 对应0号人物。


# 测试集
test_set = ImageFolder(root=os.path.join(path_init, 'test'),
                       transform=transforms.Compose([
                           transforms.CenterCrop(size=(300, 300)),
                           # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                           transforms.ToTensor()
                       ]))

# 查看测试集图片总数：
print(len(test_set))

'''
4.建立ResNet34模型
'''
# 4.1.ResNet简介
# ResNet34，ResNet50/101/152
# residual block结构
# ResNet34结构


# 4.2.构建模型
class ResidualBlock(nn.Module):
    """
    Residual Block
    """

    def __init__(self, in_channel, out_channel, stride=1, shortcut=None):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(num_features=out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=out_channel)
        )
        self.right = shortcut  # sth like nn.Module

    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        return f.relu(out)  # 加上residual后，再ReLU


class ResNet34(nn.Module):
    """
    ResNet34
    包含5个layer
    """

    def __init__(self, num_target_classes):
        super(ResNet34, self).__init__()
        # 最前面的卷积层和池化层
        self.pre = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # 5个layer，分别包含3、4、6、3个Residual Block
        self.layer1 = self.make_layer(64, 64, 3)
        self.layer2 = self.make_layer(64, 128, 4, stride=2)
        self.layer3 = self.make_layer(128, 256, 6, stride=2)
        self.layer4 = self.make_layer(256, 512, 3, stride=2)

        # 最后的全连接层
        self.fc = nn.Linear(in_features=512, out_features=num_target_classes)

        # 训练和测试信息
        self.epoch = 0
        self.len_test = None
        self.len_train = None
        self.time_epoch = []
        self.loss_epoch = []
        self.labels = None

    def make_layer(self, in_channel, out_channel, num_blocks, stride=1):
        """
        构建layer，包含多个block
        """
        shortcut = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(num_features=out_channel)
        )
        layers = [ResidualBlock(in_channel, out_channel, stride, shortcut)]  # 第一个block包含shortcut
        for i in range(1, num_blocks):
            layers.append(ResidualBlock(out_channel, out_channel))  # 之后的block不含shortcut

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = f.avg_pool2d(x, 7)  # 池化层
        x = x.view(-1, 512)

        return self.fc(x)

    def fit(self, dataset, batch_size, criterion, optimizer, epoch, device):
        """
        训练网络。
        :param dataset: such as Dataset ImageFolder
        :param batch_size: for DataLoader
        :param criterion: loss
        :param optimizer: 优化器，更新参数
        :param epoch: 遍历数据集的次数
        :param device: tenser所在设备
        :return: None
        """
        # self.train()  # 不注释掉，内存占用将会持续增加
        self.to(device)
        self.len_train = len(dataset)
        self.epoch += epoch
        data_loader = DataLoader(dataset=dataset,
                                 batch_size=batch_size,
                                 shuffle=True)
        for epo in range(epoch):
            t10 = time.time()  # 单个epoch的时长
            loss = 0.
            for images, labels in tqdm(data_loader, desc=f'epoch {epo + 1}/{epoch}'):
                # 转移至GPU，或无变化
                images = images.to(device)
                labels = labels.to(device)
                images.requires_grad = True
                # 梯度清零
                optimizer.zero_grad()
                # forward
                outputs = self.forward(images)
                # 计算损失
                loss_running = criterion(outputs, labels)
                # 反向传播
                loss_running.backward()
                # 更新参数
                optimizer.step()
                # 累加损失
                loss += loss_running.item()
            t11 = time.time()
            self.time_epoch.append(t11 - t10)
            self.loss_epoch.append(loss / len(data_loader))
        print('\nFinished.')

    def predict(self, dataset, batch_size, device):
        """
        测试网络。
        :param dataset: such as Dataset ImageFolder
        :param batch_size: for DataLoader
        :param device: tenser所在设备
        :return: None
        """
        self.to(device)
        self.len_test = len(dataset)  # 测试集的照片总数
        data_loader = DataLoader(dataset=dataset,
                                 batch_size=batch_size,
                                 shuffle=True)
        label_truth = []
        label_pred = []
        with t.no_grad():  # 测试时无需求导，可暂时关闭autograd，提高速度，节约内存
            for images, labels in tqdm(data_loader, desc='Running'):
                # 转移至GPU，或无变化
                images = images.to(device)
                labels = labels.to(device)
                # 计算图片在每个类别上的分数
                outputs = self.forward(images)
                # 预测结果是得分最高的类别
                _, predicted = t.max(outputs.data, 1)
                # 记录信息
                label_truth.extend(labels.cpu().numpy().tolist())
                label_pred.extend(predicted.cpu().numpy().tolist())
        self.labels = pd.DataFrame({'truth': label_truth, 'pred': label_pred})
        print('\nFinished.')

    def plot_train(self, style_ggplot=False):
        if self.time_epoch:
            plt.figure()
            if style_ggplot:
                plt.style.use('ggplot')
            plt.subplot(2, 1, 1)
            plt.plot(self.time_epoch)
            plt.xlabel(xlabel=None)
            plt.ylabel('Time(s)')
            # plt.xticks(ticks=range(self.epoch), labels=range(1, self.epoch + 1))
            plt.ylim([min(self.time_epoch) - .5, max(self.time_epoch) + .5])
            plt.title('Training Process')
            plt.grid(True)
            plt.subplot(2, 1, 2)
            plt.plot(self.loss_epoch)
            plt.xlabel('epoch')
            plt.ylabel('Average Loss')
            # plt.xticks(ticks=range(self.epoch), labels=range(1, self.epoch + 1))
            plt.grid(True)

            plt.tight_layout()
            plt.show()
        else:
            print('Use self.fit() first.')

    def confusion_matrix(self):
        if self.labels is not None:
            return confusion_matrix(self.labels.iloc[:, 0], self.labels.iloc[:, 1])
        else:
            print('Use self.predict() first.')

    def accuracy(self):
        if self.labels is not None:
            return np.diag(self.confusion_matrix()).sum() / self.len_test
        else:
            print('Use self.predict() first.')


clf_resnet34 = ResNet34(114)

# training
clf_resnet34.train()
clf_resnet34.fit(dataset=train_set,
                 batch_size=20,
                 criterion=nn.CrossEntropyLoss(),  # 交叉熵损失
                 optimizer=optim.SGD(clf_resnet34.parameters(), lr=.001, momentum=.9),  # 带有动量的SGD
                 epoch=10,
                 device=device)

clf_resnet34.plot_train()
print(clf_resnet34.epoch)
print(clf_resnet34.len_train)
print(clf_resnet34.time_epoch)
print(clf_resnet34.loss_epoch)

# testing
clf_resnet34.eval()
clf_resnet34.predict(dataset=test_set,
                     batch_size=20,
                     device=device)

clf_resnet34.accuracy()
clf_resnet34.confusion_matrix()
print(clf_resnet34.len_test)
print(clf_resnet34.labels)

'''
model persistence
'''
# save
t.save(clf_resnet34.state_dict(), os.path.join(os.getcwd(), '/ResNet34.pt'))
# load
clf_resnet34 = ResNet34(114)
clf_resnet34.load_state_dict(t.load(os.path.join(os.getcwd(), '/ResNet34.pt'),
                                    map_location=device))
