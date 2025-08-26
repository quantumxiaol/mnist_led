# MNIST Cpp ver

使用Eigen库+STL实现的MNIST手写数字识别

使用全连接前馈神经网络（多层感知机MLP），暂时不是LeNet

输入层 (784) → 隐藏层 (128, ReLU) → 输出层 (10, Softmax)


## Compile
```bash
# macOS: brew install eigen
# Linux: sudo apt-get install libeigen3-dev

mkdir -p output
#MacOS
clang++ -std=c++17 -O2 -I/opt/homebrew/include/eigen3 \
  cppcode/src/neural_network.cpp cppcode/src/mnist_loader.cpp cppcode/src/main.cpp \
  -o output/mnist_lenet
#Linux
g++ -std=c++17 -O2 -I/usr/include/eigen3 \
   cppcode/src/neural_network.cpp cppcode/src/mnist_loader.cpp cppcode/src/main.cpp \
   -o output/mnist_lenet

# Run
./output/mnist_lenet
```

## Result

C++版本运行结果如下：
```
Epoch 40/40, Loss: 0.0279225, Accuracy: 0.991
Training completed in 2 seconds
Final test accuracy: 87%
```

Python版本运行结果如下：
```
Epoch [40/40], Loss: 0.006997, Accuracy: 100.0%
Training complete.
Model weights saved to ./saved_weights/simple_nn_mnist.pth
Evaluating model on test set...
Evaluation complete. Time: 0.45s
Test Loss: 0.068907
Test Accuracy: 9805/10000 (98.0%)

=== Training Summary ===
Total Epochs: 40
Total Training Time: 127.72s
Average Time per Epoch: 3.19s
Total Evaluation Time: 0.45s
Final Test Accuracy: 98.0%
Model saved to: ./saved_weights/simple_nn_mnist.pth
```
相比Python版本，C++版本在训练和测试过程中速度更快，准确稍低，具体原因再看看。