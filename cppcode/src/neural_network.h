#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <Eigen/Dense>
#include <vector>
#include <random>
#include <iostream>

class SimpleNeuralNetwork {
private:
    // 网络结构参数
    int input_size;
    int hidden_size;
    int output_size;
    
    // 权重和偏置
    Eigen::MatrixXd W1, W2;
    Eigen::VectorXd b1, b2;
    
    // 随机数生成器
    std::default_random_engine rng;
    
    // 激活函数
    Eigen::VectorXd relu(const Eigen::VectorXd& x);
    Eigen::VectorXd softmax(const Eigen::VectorXd& x);
    
    // 激活函数导数
    Eigen::VectorXd relu_derivative(const Eigen::VectorXd& x);
    
public:
    SimpleNeuralNetwork(int input_sz, int hidden_sz, int output_sz);
    ~SimpleNeuralNetwork() = default;
    
    // 前向传播
    Eigen::VectorXd forward(const Eigen::VectorXd& input);
    
    // 训练函数
    void train(const std::vector<Eigen::VectorXd>& train_images,
               const std::vector<Eigen::VectorXd>& train_labels,
               int epochs, double learning_rate);
    
    // 预测函数
    int predict(const Eigen::VectorXd& image);
    
    // 评估函数
    double evaluate(const std::vector<Eigen::VectorXd>& test_images,
                   const std::vector<Eigen::VectorXd>& test_labels);
};

#endif