# 通过OpenFace实现人脸识别
本文中将介绍OpenFace中实现人脸识别的基本流程，它的结构如下图所示：
![image](https://raw.githubusercontent.com/cmusatyalab/openface/master/images/summary.jpg)

## 1. 创建人脸数据集合
- 在 openface 文件中使用mkdir命令建立一个名为./training-images/的文件夹

- 在training-images/的文件夹中，为想识别的每个人建立一个子文件夹

- 把每个人的照片拷贝到对应每个人的文件夹中

## 2.进行姿势检测和校准：
#### 2.1描述
- 输入：原始图像

- 输出：“校准”过的只含有人脸的图像

这一步要做的事情就是要检测人脸中的关键点，然后根据这些关键点对人脸做对齐校准。所谓关键点，就是下图所示的绿色的点，通常是眼角的位置、鼻子的位置、脸的轮廓点等等。有了这些关键点后，我们就可以把人脸“校准”，或者说是“对齐”。解释就是原先人脸可能比较歪，这里根据关键点，使用仿射变换将人脸统一“摆正”，尽量去消除姿势不同带来的误差。这一步我们一般叫Face Alignment。

![image](http://www.chinacloud.cn/upload/2017-06/170615150996622.png)


#### 2.2实现过程


将shape_predictor_68_face_landmarks.dat拷贝到openface\models\dlib文件夹下


```
./util/align-dlib.py ./training-images/ align outerEyesAndNose ./aligned-images/ --size 96
```
- ./training-images/表示训练的数据集合名称
- align表示主函数执行align模式，详见代码
- outerEyesAndNose表示面部特征点的提取模式
- ./aligned-images/表示创建一个子文件夹，存放带有每一个测试图像的裁剪过的并且对齐的版本
- --size 96表示默认图片大小

#### 2.3相关理论

- 问题：姿势以及面部的突出问题

由于面部朝向不同方向对计算机而言看起来完全不同，为了处理这种情况，我们将尝试打包每一个图片，以便于眼睛和嘴巴总是处于图片中的同样位置。这会使我们在下一步中更容易比较面部。

- 解决： Vahid Kazemi 和 Josephine Sullivan 在 2014 年创造的面部标志估算（face landmark estimation）的算法，使用shape_predictor_68_face_landmarks.dat作为模型进行特征点提取。

- 具体细节
1. 将提出脸上存在的 68 个特殊点（被称为 landmarks）：下巴的顶部、眼睛的外沿、眉毛的内沿等等。
2.  将简单的旋转、按比例放大或缩小以及修改图像，以便于眼睛和嘴巴最好能够居中。
- 相关文献
> One Millisecond Face Alignment with an Ensemble of Regression Trees

## 3.从对齐的图像中生成测量指标
#### 3.1描述
- 输入：校准后的单张人脸图像
- 输出：一个向量表示。

这一步就是使用深度卷积网络，将输入的人脸图像，转换成一个向量的表示。在OpenFace中使用的向量是128x1的，也就是一个128维的向量。
#### 3.2实现过程
```
./batch-represent/main.lua -outDir ./generated-embeddings/ -data ./aligned-images/
```
- -outDir ./generated-embeddings/表示测度生成后存放的子文件夹
- -data ./aligned-images/表示运行的数据所在文件夹

运行完后，这个./generated-embeddings/子文件夹会包含一个带有每张图像的嵌入的 csv 文件
#### 3.2 相关理论
- 测量指标问题

对人而言，进行人脸识别时，我们需要从每张脸中测量耳、鼻子的大小、眼睛的颜色等等。

对计算机而言，计算机只看图像中的单个像素，上述指标没有什么意义。最好的方法是让计算机自己去测量它要收集的数据。

- 测量指标解决：
深度学习要比人类更善于搞清楚面部什么部位测量起来更重要。解决方案是训练一个深度卷积神经网络。但并不是训练这它去识别照片中的目标，而是要训练它产生每张脸的 128 个测量。

这个训练过程是同时看 3 张人脸照片：
1. 加载一张已知人的训练面部图像
2. 加载另一张同一人的照片
3. 加载一张完全不同的人的照片

然后这个算法检查它当前为这三张人脸照片生成的测量数据。然后略微调整一下这个神经网络，确保#1 和#2的测量结果稍稍接近，而 #2 和 #3 的测量结果稍稍相远。

- 相关文献
FaceNet: A Unified Embedding for Face Recognition and Clustering
- [FaceNet--Google的人脸识别](http://blog.csdn.net/stdcoutzyx/article/details/46687471)

## 4.训练自己的面部检测模型
#### 4.1描述

剩下的问题就非常简单了。因为这种表示具有相同人对应的向量的距离小，不同人对应的向量距离大的特点。接下来一般的应用有以下几类：

- 人脸验证(Face Identification)。就是检测A、B是否是属于同一个人。只需要计算向量之间的距离，设定合适的报警阈值(threshold)即可。
- 人脸识别(Face Recognition)。这个应用是最多的，给定一张图片，检测数据库中与之最相似的人脸。显然可以被转换为一个求距离的最近邻问题。
- 人脸聚类(Face Clustering)。在数据库中对人脸进行聚类，直接K-Means即可。

我们所做的就是在已知人的数据库中找到这个人，这个人的测量数据要最接近我们的测试图像。在此，可以用任何基本的机器学习分类算法来做这个事情，这里使用一个简单的线性 SVM 分类器 ，但其他很多分类算法也都可以拿来用。

我们要做的就是训练一个分类器，它能吸收新测试图像的测量数据，并在已知的人中分辨出哪一个是最匹配的。运行这一分类器需要数毫秒。分类的结果就是这个人的名字。

### 4.1实现过程

```
./demos/classifier.py train ./generated-embeddings/
```

- train表示在主函数中运行train模式
- ./generated-embeddings/表示上一步生成的测度csv文件所在目录

运行结束在./generated-embeddings/目录下将产生classifier.pkl的新文件名，这个文件有你将用来识别新面部的 SVM 模型。


## 5.识别面部
#### 5.1描述

获取一张未知脸的新照片。把它像这样传递给分类器脚本，并实现分类预测。
#### 5.2实现过程
```
./demos/classifier.py infer ./generated-embeddings/classifier.pkl xxx.jpg
```
- infer表示在主函数中运行infer模式
- ./generated-embeddings/classifier.pkl表示上一步生成的训练模型
- xxx.jpg表示未知脸的文件


## 6.得到预测结果

```
=== /test-images/xxx.jpg ===
Predict yyy with m confidence.
```

- xxx.jpg表示测试的文件
- yyy表示预测结果人名
- m代表可信度
注：训练样本越多，测试样本表情越丰富，测试的可信度越高

## 7.补充功能（人脸检测）
如果希望在最原始的图像识别，或者在拍摄照片的过程中能够进行人脸检测，效果如下图：
![image](http://img.blog.csdn.net/20160726105951312)

关键技术在于图像明暗程度问题
- 问题：由于同一个人非常暗的图片和非常亮的图片将有完全不同的像素值，但通过只考虑 亮度变化的方向，暗或亮的照片将有同样准确的表征，这解决起来问题更加的容易。

- 解决：2005 年创造的方法，名为HOG（Histogram of Oriented Gradients），是一种一种快速面部检测方法。为了找到图像中的脸：

- 具体细节
1. 开始把图片变成黑白的，因为我们不需要颜色数据。
2. 挨个查看图片中的单像素（single pixel）。目标是要搞清相比于直接围绕着它的这些像素，这个像素有多暗。然后，我们想要绘制一个箭头，显示图像变暗的方向
3. 你在图片中的每个单像素重复这一过程，最终每个像素都会被箭头取代。这些箭头被称为梯度，它们显示了整张图片中由亮变暗的过程。
   
但是，保留每个单像素的梯度给了我们太多的细节。如果我们只看更高层次的亮度/暗度基础流这样会更好，我们就能看到图片的基础模式。

为了做到这一点，我们将图片分割成 16×16的像素方块。在每个方块下，我们将计算每个主要方向中有多少个梯度点（多少指向上、多少指向下、指向右，等等）。然后，我们将用指向最多的方向箭头取代这个方块。

最终结果是我们以一种非常简单的方式，把原始图片变成了一个能抓住面部基础结构的表征