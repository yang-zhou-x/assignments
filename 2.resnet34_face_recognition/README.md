
## 基于残差网络ResNet34的多角度人脸识别

本文使用 **Pytorch** 建立 **ResNet34** ，对114人的多角度人像进行分类预测，即输入测试图片后返回人物的对应标签。  
代码在[code.py](https://github.com/yang-zhou-x/assignments/blob/master/2.resnet34_face_recognition/code.py)

每个人包含7张照片：左侧脸照、45°照、正面照、135°照、右侧脸照、正面照（不露齿笑）和正面照（露齿笑），极少数存在不足7张或超过7张的情况。
每人随机抽取2张照片作为测试集，预测准确率为82.28%。
  
本文的难点在于预测时人像角度的不确定性，如根据正面照来识别侧面照，或是根据侧面照来识别正面照。

代码兼容CPU和GPU。更详细的内容在代码注释中。

## 数据来源
CVL Face Database by Peter Peer, University of Ljubljana  
  
详细信息：
![IMG_0847.JPG](https://i.loli.net/2019/03/02/5c7a78af9590e.jpg)

## 环境配置
CUDA 9.2  
cuDNN 7.3.1  
PyTorch 0.4.1

## 作者
周扬，中国人民大学
