# MNIST LED Display

## TODO

我想要通过硬件展示一个神经网络的运行。

我计划使用MNIST数据集，运行最简单的LeCNN，然后制作LED板子，一个灯珠代表一个神经元，激活亮，不激活不亮，最后一层层走到10个标签，代表0-9。

前面有28*28的LED屏幕，对应MNIST的数据集的图像大小。

## Train & Eval Result

    === Training Summary ===
    Total Epochs: 20
    Total Training Time: 114.38s
    Average Time per Epoch: 5.72s
    Total Evaluation Time: 0.39s
    Final Test Accuracy: 99.17%
    Model saved to: ./saved_weights/lenet_mnist.pth

