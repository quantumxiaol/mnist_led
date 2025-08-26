#include "neural_network.h"
#include <cmath>
#include <algorithm>
#include <iostream>

SimpleNeuralNetwork::SimpleNeuralNetwork(int input_sz, int hidden_sz, int output_sz)
    : input_size(input_sz), hidden_size(hidden_sz), output_size(output_sz), rng(42) {
    
    // Xavier初始化
    double scale1 = sqrt(2.0 / input_sz);
    double scale2 = sqrt(2.0 / hidden_sz);
    
    // 初始化权重
    W1 = Eigen::MatrixXd::Random(hidden_sz, input_sz) * scale1;
    W2 = Eigen::MatrixXd::Random(output_sz, hidden_sz) * scale2;
    
    // 初始化偏置
    b1 = Eigen::VectorXd::Zero(hidden_sz);
    b2 = Eigen::VectorXd::Zero(output_sz);
}

Eigen::VectorXd SimpleNeuralNetwork::relu(const Eigen::VectorXd& x) {
    return x.array().max(0.0);
}

Eigen::VectorXd SimpleNeuralNetwork::relu_derivative(const Eigen::VectorXd& x) {
    Eigen::VectorXd result = Eigen::VectorXd::Zero(x.size());
    for (int i = 0; i < x.size(); ++i) {
        result(i) = (x(i) > 0) ? 1.0 : 0.0;
    }
    return result;
}

Eigen::VectorXd SimpleNeuralNetwork::softmax(const Eigen::VectorXd& x) {
    Eigen::VectorXd exp_x = (x.array() - x.maxCoeff()).exp();
    return exp_x / exp_x.sum();
}

Eigen::VectorXd SimpleNeuralNetwork::forward(const Eigen::VectorXd& input) {
    // 第一层：输入 -> 隐藏层
    Eigen::VectorXd z1 = W1 * input + b1;
    Eigen::VectorXd a1 = relu(z1);
    
    // 第二层：隐藏层 -> 输出层
    Eigen::VectorXd z2 = W2 * a1 + b2;
    Eigen::VectorXd a2 = softmax(z2);
    
    return a2;
}

void SimpleNeuralNetwork::train(const std::vector<Eigen::VectorXd>& train_images,
                               const std::vector<Eigen::VectorXd>& train_labels,
                               int epochs, double learning_rate) {
    
    int num_samples = train_images.size();
    
    for (int epoch = 0; epoch < epochs; ++epoch) {
        double total_loss = 0.0;
        int correct = 0;
        
        for (size_t i = 0; i < train_images.size(); ++i) {
            const Eigen::VectorXd& input = train_images[i];
            const Eigen::VectorXd& label = train_labels[i];
            
            // 前向传播
            Eigen::VectorXd z1 = W1 * input + b1;
            Eigen::VectorXd a1 = relu(z1);
            Eigen::VectorXd z2 = W2 * a1 + b2;
            Eigen::VectorXd a2 = softmax(z2);
            
            // 计算损失
            double loss = 0.0;
            for (int j = 0; j < output_size; ++j) {
                if (label(j) > 0) {
                    loss -= log(std::max(a2(j), 1e-15));
                }
            }
            total_loss += loss;
            
            // 计算准确率
            int predicted = 0;
            int actual = 0;
            double max_pred = a2(0);
            double max_actual = label(0);
            
            for (int j = 1; j < output_size; ++j) {
                if (a2(j) > max_pred) {
                    max_pred = a2(j);
                    predicted = j;
                }
                if (label(j) > max_actual) {
                    max_actual = label(j);
                    actual = j;
                }
            }
            if (predicted == actual) correct++;
            
            // 反向传播
            // 输出层梯度
            Eigen::VectorXd dz2 = a2 - label;
            
            // 隐藏层梯度
            Eigen::VectorXd da1 = W2.transpose() * dz2;
            Eigen::VectorXd dz1 = da1.array() * relu_derivative(z1).array();
            
            // 更新权重和偏置
            W2 -= learning_rate * (dz2 * a1.transpose());
            b2 -= learning_rate * dz2;
            W1 -= learning_rate * (dz1 * input.transpose());
            b1 -= learning_rate * dz1;
        }
        
        if ((epoch + 1) % 1 == 0) {
            std::cout << "Epoch " << epoch + 1 << "/" << epochs 
                      << ", Loss: " << total_loss / num_samples
                      << ", Accuracy: " << static_cast<double>(correct) / num_samples
                      << std::endl;
        }
    }
}

int SimpleNeuralNetwork::predict(const Eigen::VectorXd& image) {
    Eigen::VectorXd output = forward(image);
    int predicted = 0;
    double max_val = output(0);
    for (int i = 1; i < output_size; ++i) {
        if (output(i) > max_val) {
            max_val = output(i);
            predicted = i;
        }
    }
    return predicted;
}

double SimpleNeuralNetwork::evaluate(const std::vector<Eigen::VectorXd>& test_images,
                                    const std::vector<Eigen::VectorXd>& test_labels) {
    int correct = 0;
    for (size_t i = 0; i < test_images.size(); ++i) {
        int predicted = predict(test_images[i]);
        int actual = 0;
        for (int j = 0; j < test_labels[i].size(); ++j) {
            if (test_labels[i](j) == 1) {
                actual = j;
                break;
            }
        }
        if (predicted == actual) correct++;
    }
    return static_cast<double>(correct) / test_images.size();
}