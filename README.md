

# 目标检测和跟踪学习

[toc]



## 目标检测

1. **define** ： 在对象周围创建边界框来分类和检测对象

2. **方法**：基于神经网络(基于深度学习）或非神经方法（基于经典机器学习）

   后者包括 ： viola-jones目标检测，SIFT，定向梯度直方图等



**RCNN(region+CNN)**：

Ross Girshick 等人在 2014 年的论文中描述了 R-CNN。来自加州大学伯克利分校的题为 “[Rich feature hierarchies for accurate object detection and semantic segmentation](https://arxiv.org/abs/1311.2524).”（丰富特征层次结构）

CNN :将大数据量图片降维成小数据量，且保留图像原有特征（类似视觉，即使图像翻转，旋转，变换位置也能有效识别）

传统方式，同样的内容，比如旋转，参数就发生很大变化

![R-CNN 模型架构总结](https://machinelearningmastery.com/wp-content/uploads/2019/03/Summary-of-the-R-CNN-Model-Architecture.png)

RCNN 3大模块

- **Module 1: Region Proposal**. Generate and extract category independent region proposals, e.g. candidate bounding boxes.
- **Module 2: Feature Extractor**. Extract feature from each candidate region, e.g. using a deep convolutional neural network.
- **Module 3: Classifier**. Classify features as one of the known class, e.g. linear SVM classifier model.

### 卷积神经网络-CNN 的基本原理

**典型CNN组成**：

1. 卷积层（提取局部特征）
2. 池化层（大幅降低参数级，降维）
3. 全连接层（输出结果）





#### 卷积——提取特征

1. ![卷积层运算过程](https://easyai.tech/wp-content/uploads/2022/08/f144f-2019-06-19-juanji.gif)

卷积层的运算过程如上，这个过程我们可以理解为我们使用一个过滤器（卷积核）来过滤图像的各个小区域，从而得到这些小区域的特征值



#### 池化层（下采样）——数据降维，避免过拟合

![池化层过程](https://easyai.tech/wp-content/uploads/2022/08/3fd53-2019-06-19-chihua.gif)





**池化层相比卷积层可以更有效的降低数据维度，这么做不但可以大大减少运算量，还可以有效的避免过拟合**





#### 全连接层——输出结果

![全连接层](https://easyai.tech/wp-content/uploads/2022/08/c1a6d-2019-06-19-quanlianjie.png)



**Region的发展**：

1. 起初通过窗口扫描,但初始窗口的尺寸需要多次尝试等等，很耗时间 

 	2. 改成 SS（selective search）选择性搜索，通过纹理，颜色等将视图划分，再确定候选区域，进行分类处理.



Fast-RCNN

1. 将图片输入到CNN得到特征图

 	2. 划分特征图的SS区域得到特征框，在ROI池化层将每个特征框池化到统一大小
 	3. 最后将特征框输入到全连接层进行分类和回归

Faster-RCNN![Faster R-CNN 模型架构总结](https://machinelearningmastery.com/wp-content/uploads/2019/03/Summary-of-the-Faster-R-CNN-Model-Architecture.png)

1. 变化  SS变为RPN

   RPN原理：

   1. 为输入特征图的每个像素点生成9个候选框
   
    	2. 对生成的基础候选框做修正处理，删除不包含目标的候选框
    	3. 对超出图像边界的候选框做 裁剪
    	4. 忽略长或宽太小的候选框
    	5. 对当前候选框做得分高低排序，选取前12000个候选框
    	6. 排除重叠框
    	7. 选前2000个做二次修正

### YOLO

YOLO 模型首先由 Joseph Redmon 等人描述。在 2015 年题为“ [You Only Look Once: ](https://arxiv.org/abs/1506.02640)[Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640) ”的论文中。

该方法涉及一个端到端训练的单个神经网络，该网络将照片作为输入并直接预测每个边界框的边界框和类标签。

### YOLO家族

## 目标跟踪

1. define：是一个深度学习过程（估计预测运动对象），跟踪对象的运动
2. two ways:
   1. 密集跟踪(a series detections) :处理所有快照的像素进行计算,进行一系列的检测
   2. 稀疏跟踪(estimation + common sense)：试图预测轨迹，进行动态检测
3. 过程：
   1. 定义目标
   2. 外观建模（目标的移动在不同的场景中(光照条件不同，角度等）可能会改变外观)
   3. 运动估计
   4. 目标定位
4. 级别： 单目标跟踪（SOT）和多目标跟踪（MOT）
    1. 前者指跟踪单个类的对象
    2. 后者指跟踪每个感兴趣的对象
5. 挑战和解决方案
   1. 遮挡（可能会是刚开始识别的对象被跟踪为新对象），可以进行occlusion sensitivity，允许识别对象的哪个特定特征正混淆网络，识别后，相似的图像来纠正偏差。
   2. 背景，难干扰以提取特征，检测和跟踪。可以使用背景稀疏的策划好的数据集
   3. 训练和跟踪速度：不仅仅能准确执行检测和定位，还要尽可能短的时间完成。 CNN修改
   4. 多空间尺度：要跟踪的对象可以有各种大小和纵横比，可能会混淆算法
      1. 锚框
      2. 特征图：CNN捕获输入图像的输出
      3. 图像和特征金字塔表示
6. MDNet：多域网络利用大规模数据进行训练的算法
7. CNN

# 深度学习

将世界表示为嵌套的层次概念体系（由简单概念之间的联系定义复杂概念、从一般抽象概括到高级抽象表示）