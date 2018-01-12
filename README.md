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

#### Google TensorFlow Playgound:
![google_tensorflow_playground_1.png](https://github.com/JenifferWuUCLA/TensorFlow-Google-Projects/blob/master/images/google_tensorflow_playground_1.png)
![google_tensorflow_playground_２.png](https://github.com/JenifferWuUCLA/TensorFlow-Google-Projects/blob/master/images/google_tensorflow_playground_2.png)

---

> Tensorflow-Google-Projects# python forward-propagation-algorithm.py 

[[ 3.95757794]

 [ 1.15376544]
 
 [ 3.16749191]]

---

> Tensorflow-Google-Projects# python neural_networks_classification.py 

[[-0.81131822  1.48459876  0.06532937]

 [-2.44270396  0.0992484   0.59122431]]
 
[[-0.81131822]

 [ 1.48459876]
 
 [ 0.06532937]]
 
> After 0 training step(s), cross entropy on all data is 0.0674925

> After 1000 training step(s), cross entropy on all data is 0.0163385

> After 2000 training step(s), cross entropy on all data is 0.00907547

> After 3000 training step(s), cross entropy on all data is 0.00714436

> After 4000 training step(s), cross entropy on all data is 0.00578471

[[-1.9618274   2.58235407  1.68203783]

 [-3.4681716   1.06982327  2.11788988]]
 
[[-1.8247149 ]

 [ 2.68546653]
 
 [ 1.41819501]]

---

Tensorflow-Google-Projects# python MNIST_data_preprocessing.py 

Successfully downloaded train-images-idx3-ubyte.gz 9912422 bytes.

Extracting /home/jenifferwu/TensorFlow_data/MNIST_data/train-images-idx3-ubyte.gz

Successfully downloaded train-labels-idx1-ubyte.gz 28881 bytes.

Extracting /home/jenifferwu/TensorFlow_data/MNIST_data/train-labels-idx1-ubyte.gz

Successfully downloaded t10k-images-idx3-ubyte.gz 1648877 bytes.

Extracting /home/jenifferwu/TensorFlow_data/MNIST_data/t10k-images-idx3-ubyte.gz

Successfully downloaded t10k-labels-idx1-ubyte.gz 4542 bytes.

Extracting /home/jenifferwu/TensorFlow_data/MNIST_data/t10k-labels-idx1-ubyte.gz

Training data size:  55000

Validating data size:  5000

Testing data size:  10000

> Example training data:  [ 0.          0.          0.          0.          0.          0.          0.

>   0.          0.          0.          0.          0.          0.          0.

>   0.          0.          0.          0.          0.          0.          0.

>   0.          0.          0.          0.          0.          0.          0.

>   0.          0.          0.          0.          0.          0.          0.

>   0.          0.          0.          0.          0.          0.          0.

>   0.          0.          0.          0.          0.          0.          0.

>   0.          0.          0.          0.          0.          0.          0.

>   0.          0.          0.          0.          0.          0.          0.

>   0.          0.          0.          0.          0.          0.          0.

>   0.          0.          0.          0.          0.          0.          0.

>   0.          0.          0.          0.          0.          0.          0.

>   0.          0.          0.          0.          0.          0.          0.

>   0.          0.          0.          0.          0.          0.          0.

>   0.          0.          0.          0.          0.          0.          0.

>   0.          0.          0.          0.          0.          0.          0.

>   0.          0.          0.          0.          0.          0.          0.

>   0.          0.          0.          0.          0.          0.          0.

>   0.          0.          0.          0.          0.          0.          0.

>   0.          0.          0.          0.          0.          0.          0.

>   0.          0.          0.          0.          0.          0.          0.

>   0.          0.          0.          0.          0.          0.          0.

>   0.          0.          0.          0.          0.          0.          0.

>   0.          0.          0.          0.          0.          0.          0.

>   0.          0.          0.          0.          0.          0.          0.

>   0.          0.          0.          0.          0.          0.          0.

>   0.          0.          0.          0.          0.          0.          0.

>   0.          0.          0.          0.          0.          0.          0.

>   0.          0.          0.          0.          0.          0.          0.

>   0.          0.          0.          0.          0.38039219  0.37647063

>   0.3019608   0.46274513  0.2392157   0.          0.          0.          0.

>   0.          0.          0.          0.          0.          0.          0.

>   0.          0.          0.          0.          0.35294119  0.5411765

>   0.92156869  0.92156869  0.92156869  0.92156869  0.92156869  0.92156869

>   0.98431379  0.98431379  0.97254908  0.99607849  0.96078438  0.92156869

>   0.74509805  0.08235294  0.          0.          0.          0.          0.

>   0.          0.          0.          0.          0.          0.

>   0.54901963  0.98431379  0.99607849  0.99607849  0.99607849  0.99607849

>   0.99607849  0.99607849  0.99607849  0.99607849  0.99607849  0.99607849

>   0.99607849  0.99607849  0.99607849  0.99607849  0.74117649  0.09019608

>   0.          0.          0.          0.          0.          0.          0.

>   0.          0.          0.          0.88627458  0.99607849  0.81568635

>   0.78039223  0.78039223  0.78039223  0.78039223  0.54509807  0.2392157

>   0.2392157   0.2392157   0.2392157   0.2392157   0.50196081  0.8705883

>   0.99607849  0.99607849  0.74117649  0.08235294  0.          0.          0.

  0.          0.          0.          0.          0.          0.

  0.14901961  0.32156864  0.0509804   0.          0.          0.          0.

  0.          0.          0.          0.          0.          0.          0.

  0.13333334  0.83529419  0.99607849  0.99607849  0.45098042  0.          0.

  0.          0.          0.          0.          0.          0.          0.

  0.          0.          0.          0.          0.          0.          0.

  0.          0.          0.          0.          0.          0.          0.

  0.          0.32941177  0.99607849  0.99607849  0.91764712  0.          0.

  0.          0.          0.          0.          0.          0.          0.

  0.          0.          0.          0.          0.          0.          0.

  0.          0.          0.          0.          0.          0.          0.

  0.          0.32941177  0.99607849  0.99607849  0.91764712  0.          0.

  0.          0.          0.          0.          0.          0.          0.

  0.          0.          0.          0.          0.          0.          0.

  0.          0.          0.          0.          0.          0.          0.

  0.41568631  0.6156863   0.99607849  0.99607849  0.95294124  0.20000002

  0.          0.          0.          0.          0.          0.          0.

  0.          0.          0.          0.          0.          0.          0.

  0.          0.          0.          0.09803922  0.45882356  0.89411771

  0.89411771  0.89411771  0.99215692  0.99607849  0.99607849  0.99607849

  0.99607849  0.94117653  0.          0.          0.          0.          0.

  0.          0.          0.          0.          0.          0.          0.

  0.          0.          0.          0.26666668  0.4666667   0.86274517

  0.99607849  0.99607849  0.99607849  0.99607849  0.99607849  0.99607849

  0.99607849  0.99607849  0.99607849  0.55686277  0.          0.          0.

  0.          0.          0.          0.          0.          0.          0.

  0.          0.          0.          0.14509805  0.73333335  0.99215692

  0.99607849  0.99607849  0.99607849  0.87450987  0.80784321  0.80784321

  0.29411766  0.26666668  0.84313732  0.99607849  0.99607849  0.45882356

  0.          0.          0.          0.          0.          0.          0.

  0.          0.          0.          0.          0.          0.44313729

  0.8588236   0.99607849  0.94901967  0.89019614  0.45098042  0.34901962

  0.12156864  0.          0.          0.          0.          0.7843138

  0.99607849  0.9450981   0.16078432  0.          0.          0.          0.

  0.          0.          0.          0.          0.          0.          0.

  0.          0.66274512  0.99607849  0.6901961   0.24313727  0.          0.

  0.          0.          0.          0.          0.          0.18823531

  0.90588242  0.99607849  0.91764712  0.          0.          0.          0.

  0.          0.          0.          0.          0.          0.          0.

  0.          0.          0.07058824  0.48627454  0.          0.          0.

  0.          0.          0.          0.          0.          0.

  0.32941177  0.99607849  0.99607849  0.65098041  0.          0.          0.

  0.          0.          0.          0.          0.          0.          0.

  0.          0.          0.          0.          0.          0.          0.

  0.          0.          0.          0.          0.          0.          0.

  0.54509807  0.99607849  0.9333334   0.22352943  0.          0.          0.

  0.          0.          0.          0.          0.          0.          0.

  0.          0.          0.          0.          0.          0.          0.

  0.          0.          0.          0.          0.          0.

  0.82352948  0.98039222  0.99607849  0.65882355  0.          0.          0.

  0.          0.          0.          0.          0.          0.          0.

  0.          0.          0.          0.          0.          0.          0.

  0.          0.          0.          0.          0.          0.          0.

  0.94901967  0.99607849  0.93725497  0.22352943  0.          0.          0.

  0.          0.          0.          0.          0.          0.          0.

  0.          0.          0.          0.          0.          0.          0.

  0.          0.          0.          0.          0.          0.

  0.34901962  0.98431379  0.9450981   0.33725491  0.          0.          0.

  0.          0.          0.          0.          0.          0.          0.

  0.          0.          0.          0.          0.          0.          0.


  0.          0.          0.          0.          0.          0.

  0.01960784  0.80784321  0.96470594  0.6156863   0.          0.          0.

  0.          0.          0.          0.          0.          0.          0.

  0.          0.          0.          0.          0.          0.          0.

  0.          0.          0.          0.          0.          0.          0.

  0.01568628  0.45882356  0.27058825  0.          0.          0.          0.

  0.          0.          0.          0.          0.          0.          0.

  0.          0.          0.          0.          0.          0.          0.

  0.          0.          0.          0.          0.          0.          0.

  0.          0.          0.          0.          0.          0.          0.

  0.          0.          0.          0.          0.          0.          0.        ]

Examples training data label:  [ 0.  0.  0.  0.  0.  0.  0.  1.  0.  0.]

---

Tensorflow-Google-Projects# python MNIST_handwritten_digit_recognition.py 
> Extracting /home/jenifferwu/TensorFlow_data/tmp/data/train-images-idx3-ubyte.gz
Extracting /home/jenifferwu/TensorFlow_data/tmp/data/train-labels-idx1-ubyte.gz
Extracting /home/jenifferwu/TensorFlow_data/tmp/data/t10k-images-idx3-ubyte.gz
Extracting /home/jenifferwu/TensorFlow_data/tmp/data/t10k-labels-idx1-ubyte.gz

> After 0 training step(s), validation accuracy using average model is 0.0386 
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