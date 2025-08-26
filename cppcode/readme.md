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
```
Epoch 40/40, Loss: 0.0279225, Accuracy: 0.991
Training completed in 2 seconds
Final test accuracy: 87%
```
相比Python版本，C++版本在训练和测试过程中速度更快，准确稍低（网络结构不一样）。