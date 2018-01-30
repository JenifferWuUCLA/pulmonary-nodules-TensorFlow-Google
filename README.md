# TensorFlow: Google Deep Learning Framework

## TensorFlow实现卷积神经网络与图像识别
> ##### @author Jeniffer Wu

## Overview

>#### 前向传播算法
>#### TensorFlow训练神经网络模型
>#### 深度神经网络优化算法
>#### TensorFlow的MNIST数字识别
>#### 卷积神经网络模型与迁移学习
>#### TensorFlow数字图像数据处理
>#### 循环神经网络和LTSM结构（自然语言建模、时间序列预测）
>#### TensorBoard可视化
>#### 分布式TensorFlow模型训练

---
>#### /tmp/tensorflow/mnist/logs# tensorboard --logdir=mnist_with_summaries/
Starting TensorBoard 47 at http://0.0.0.0:6006

![01.png](https://github.com/JenifferWuUCLA/TensorFlow-Google-Projects/blob/master/images/01.png)

![02.png](https://github.com/JenifferWuUCLA/TensorFlow-Google-Projects/blob/master/images/02.png)

![03.png](https://github.com/JenifferWuUCLA/TensorFlow-Google-Projects/blob/master/images/03.png)

![04.png](https://github.com/JenifferWuUCLA/TensorFlow-Google-Projects/blob/master/images/04.png)

![05.png](https://github.com/JenifferWuUCLA/TensorFlow-Google-Projects/blob/master/images/05.png)

![06.png](https://github.com/JenifferWuUCLA/TensorFlow-Google-Projects/blob/master/images/06.png)

![07.png](https://github.com/JenifferWuUCLA/TensorFlow-Google-Projects/blob/master/images/07.png)

![08.png](https://github.com/JenifferWuUCLA/TensorFlow-Google-Projects/blob/master/images/08.png)

![09.png](https://github.com/JenifferWuUCLA/TensorFlow-Google-Projects/blob/master/images/09.png)

![10.png](https://github.com/JenifferWuUCLA/TensorFlow-Google-Projects/blob/master/images/10.png)

![11.png](https://github.com/JenifferWuUCLA/TensorFlow-Google-Projects/blob/master/images/11.png)

![12.png](https://github.com/JenifferWuUCLA/TensorFlow-Google-Projects/blob/master/images/12.png)

![13.png](https://github.com/JenifferWuUCLA/TensorFlow-Google-Projects/blob/master/images/13.png)

![14.png](https://github.com/JenifferWuUCLA/TensorFlow-Google-Projects/blob/master/images/14.png)

---

>#### Tensorflow-Google-Projects# python MNIST_handwritten_digit_recognition.py

Extracting /home/jenifferwu/TensorFlow_data/tmp/data/train-images-idx3-ubyte.gz

Extracting /home/jenifferwu/TensorFlow_data/tmp/data/train-labels-idx1-ubyte.gz

Extracting /home/jenifferwu/TensorFlow_data/tmp/data/t10k-images-idx3-ubyte.gz

Extracting /home/jenifferwu/TensorFlow_data/tmp/data/t10k-labels-idx1-ubyte.gz

After 0 training step(s), validation accuracy using average model is 0.0386

After 1000 training step(s), validation accuracy using average model is 0.9772

After 2000 training step(s), validation accuracy using average model is 0.9812

After 3000 training step(s), validation accuracy using average model is 0.9828

After 4000 training step(s), validation accuracy using average model is 0.9828

After 5000 training step(s), validation accuracy using average model is 0.9836

After 6000 training step(s), validation accuracy using average model is 0.9826

After 7000 training step(s), validation accuracy using average model is 0.9842

After 8000 training step(s), validation accuracy using average model is 0.9828

After 9000 training step(s), validation accuracy using average model is 0.9838

After 10000 training step(s), validation accuracy using average model is 0.984

After 11000 training step(s), validation accuracy using average model is 0.9842

After 12000 training step(s), validation accuracy using average model is 0.9834

After 13000 training step(s), validation accuracy using average model is 0.9836

After 14000 training step(s), validation accuracy using average model is 0.9834

After 15000 training step(s), validation accuracy using average model is 0.9832

After 16000 training step(s), validation accuracy using average model is 0.9834

After 17000 training step(s), validation accuracy using average model is 0.9834

After 18000 training step(s), validation accuracy using average model is 0.984

After 19000 training step(s), validation accuracy using average model is 0.9836

After 20000 training step(s), validation accuracy using average model is 0.9836

After 21000 training step(s), validation accuracy using average model is 0.9842

After 22000 training step(s), validation accuracy using average model is 0.9838

After 23000 training step(s), validation accuracy using average model is 0.9842

After 24000 training step(s), validation accuracy using average model is 0.9838

After 25000 training step(s), validation accuracy using average model is 0.9836

After 26000 training step(s), validation accuracy using average model is 0.9838

After 27000 training step(s), validation accuracy using average model is 0.9846

After 28000 training step(s), validation accuracy using average model is 0.9846

After 29000 training step(s), validation accuracy using average model is 0.9842

After 30000 training step(s), test accuracy using average model is 0.9847


>#### Tensorflow-Google-Projects# python mnist_train.py

Extracting /home/jenifferwu/TensorFlow_data/tmp/data/train-images-idx3-ubyte.gz

Extracting /home/jenifferwu/TensorFlow_data/tmp/data/train-labels-idx1-ubyte.gz

Extracting /home/jenifferwu/TensorFlow_data/tmp/data/t10k-images-idx3-ubyte.gz

Extracting /home/jenifferwu/TensorFlow_data/tmp/data/t10k-labels-idx1-ubyte.gz

After 1 training step(s), loss on training batch is 3.07277.

After 1001 training step(s), loss on training batch is 0.261659.

After 2001 training step(s), loss on training batch is 0.187801.

After 3001 training step(s), loss on training batch is 0.154186.

After 4001 training step(s), loss on training batch is 0.122932.

After 5001 training step(s), loss on training batch is 0.106926.

After 6001 training step(s), loss on training batch is 0.10171.

After 7001 training step(s), loss on training batch is 0.0913763.

After 8001 training step(s), loss on training batch is 0.0761578.

After 9001 training step(s), loss on training batch is 0.0763629.

After 10001 training step(s), loss on training batch is 0.0703265.

After 11001 training step(s), loss on training batch is 0.0617021.

After 12001 training step(s), loss on training batch is 0.0633702.

After 13001 training step(s), loss on training batch is 0.053284.

After 14001 training step(s), loss on training batch is 0.0519821.

After 15001 training step(s), loss on training batch is 0.0521027.

After 16001 training step(s), loss on training batch is 0.047666.

After 17001 training step(s), loss on training batch is 0.0480853.

After 18001 training step(s), loss on training batch is 0.0486614.

After 19001 training step(s), loss on training batch is 0.0445071.

After 20001 training step(s), loss on training batch is 0.042046.

After 21001 training step(s), loss on training batch is 0.0400587.

After 22001 training step(s), loss on training batch is 0.0440648.

After 23001 training step(s), loss on training batch is 0.0403247.

After 24001 training step(s), loss on training batch is 0.0388441.

After 25001 training step(s), loss on training batch is 0.0382769.

After 26001 training step(s), loss on training batch is 0.042565.

After 27001 training step(s), loss on training batch is 0.0356875.

After 28001 training step(s), loss on training batch is 0.0375919.

After 29001 training step(s), loss on training batch is 0.0350133.