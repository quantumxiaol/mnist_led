// MNIST Neural Network Training
// 
// 跨平台编译说明:
// 
// === macOS ===
// 使用系统默认clang++:
// clang++ -std=c++17 -O2 -I/opt/homebrew/include/eigen3 \
   cppcode/src/neural_network.cpp cppcode/src/mnist_loader.cpp cppcode/src/main.cpp \
   -o output
//
// 使用Homebrew clang++:
// /opt/homebrew/opt/llvm/bin/clang++ -std=c++17 -O2 -I/opt/homebrew/include/eigen3 \
   cppcode/src/neural_network.cpp cppcode/src/mnist_loader.cpp cppcode/src/main.cpp \
   -o output/mnist_lenet
//
// === Linux ===
// 使用g++:
// g++ -std=c++17 -O2 -I/usr/include/eigen3 \
   cppcode/src/neural_network.cpp cppcode/src/mnist_loader.cpp cppcode/src/main.cpp \
   -o output/mnist_lenet
//
// 使用clang++:
// clang++ -std=c++17 -O2 -I/usr/include/eigen3 \
   cppcode/src/neural_network.cpp cppcode/src/mnist_loader.cpp cppcode/src/main.cpp \
   -o output/mnist_lenet
//
// === Windows (MinGW) ===
// g++ -std=c++17 -O2 -I"C:\Program Files\Eigen3\include" \
   cppcode\src\neural_network.cpp cppcode\src\mnist_loader.cpp cppcode\src\main.cpp \
   -o output/mnist_lenet.exe
//
// 使用MSVC:
// cl /EHsc /W4 /O2 /std:c++17 ^ 
//    /I "C:\Program Files\Eigen3\include" ^
//    /utf-8 ^
//    cppcode\src\neural_network.cpp cppcode\src\mnist_loader.cpp cppcode\src\main.cpp ^
//    /link /out:output\mnist_lenet.exe
//
// === 使用CMake (所有平台) ===
// mkdir build && cd build
// cmake ..
// cmake --build .
//
// 注意: 如果没有安装Eigen3，请先安装:
// macOS: brew install eigen
// Linux: sudo apt-get install libeigen3-dev
// Windows: 下载Eigen3并解压到合适目录，默认路径为C:\Program Files\Eigen3



#include "neural_network.h"
#include "mnist_loader.h"
#include <iostream>
#include <chrono>
#include <vector>
#include <random>

// 简单的MNIST数据生成器（用于测试）
std::vector<Eigen::VectorXd> generate_dummy_mnist_data(int num_samples, int size) {
    std::vector<Eigen::VectorXd> data;
    std::default_random_engine rng(42);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    
    for (int i = 0; i < num_samples; ++i) {
        Eigen::VectorXd sample = Eigen::VectorXd::Zero(size);
        for (int j = 0; j < size; ++j) {
            sample(j) = dist(rng);
        }
        data.push_back(sample);
    }
    return data;
}

std::vector<Eigen::VectorXd> generate_dummy_labels(int num_samples, int num_classes) {
    std::vector<Eigen::VectorXd> labels;
    std::default_random_engine rng(42);
    std::uniform_int_distribution<int> dist(0, num_classes - 1);
    
    for (int i = 0; i < num_samples; ++i) {
        Eigen::VectorXd label = Eigen::VectorXd::Zero(num_classes);
        int class_idx = dist(rng);
        label(class_idx) = 1.0;
        labels.push_back(label);
    }
    return labels;
}

int main() {
    std::cout << "MNIST Neural Network Training" << std::endl;
    std::cout << "=============================" << std::endl;
    
    try {
        // 创建MNIST数据加载器
        std::cout << "Loading MNIST dataset..." << std::endl;
        MNISTLoader loader("data/MNIST/raw/train-images-idx3-ubyte",
                          "data/MNIST/raw/train-labels-idx1-ubyte",
                          "data/MNIST/raw/t10k-images-idx3-ubyte",
                          "data/MNIST/raw/t10k-labels-idx1-ubyte");
        
        // 加载训练数据（使用小样本进行测试）
        std::cout << "Loading training data..." << std::endl;
        auto [train_images, train_labels] = loader.load_training_data(1000); // 只加载1000个样本测试
        
        // 加载测试数据
        std::cout << "Loading test data..." << std::endl;
        auto [test_images, test_labels] = loader.load_test_data(100); // 只加载100个样本测试
        
        std::cout << "Training data loaded: " << train_images.size() << " samples" << std::endl;
        std::cout << "Test data loaded: " << test_images.size() << " samples" << std::endl;
        
        // 网络参数
        const int input_size = 784;   // 28x28 pixels
        const int hidden_size = 128;  // 隐藏层神经元数量
        const int output_size = 10;   // 10个数字类别
        int train_epochs = 40;
        double learning_rate = 0.01;
        
        // 创建神经网络
        std::cout << "Creating neural network..." << std::endl;
        SimpleNeuralNetwork network(input_size, hidden_size, output_size);
        
        // 评估初始准确率
        std::cout << "Initial test accuracy: " 
                  << network.evaluate(test_images, test_labels) * 100 << "%" << std::endl;
        
        // 开始训练
        std::cout << "Starting training..." << std::endl;
        auto start_time = std::chrono::high_resolution_clock::now();
        
        network.train(train_images, train_labels, train_epochs, learning_rate); // 40 epochs, learning rate 0.1
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
        
        std::cout << "Training completed in " << duration.count() << " seconds" << std::endl;
        
        // 评估最终准确率
        double final_accuracy = network.evaluate(test_images, test_labels);
        std::cout << "Final test accuracy: " << final_accuracy * 100 << "%" << std::endl;
        
        // 测试几个预测
        std::cout << "\nSample predictions:" << std::endl;
        for (int i = 0; i < 5 && i < test_images.size(); ++i) {
            int predicted = network.predict(test_images[i]);
            int actual = 0;
            for (int j = 0; j < test_labels[i].size(); ++j) {
                if (test_labels[i](j) == 1) {
                    actual = j;
                    break;
                }
            }
            std::cout << "Sample " << i << ": Predicted=" << predicted 
                      << ", Actual=" << actual 
                      << (predicted == actual ? " ✓" : " ✗") << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    std::cout << "MNIST training completed successfully!" << std::endl;
    return 0;
}