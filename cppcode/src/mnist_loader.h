#ifndef MNIST_LOADER_H
#define MNIST_LOADER_H

#include <Eigen/Dense>
#include <vector>
#include <string>
#include <fstream>
#include <cstdint>

class MNISTLoader {
private:
    std::string train_images_path;
    std::string train_labels_path;
    std::string test_images_path;
    std::string test_labels_path;
    
    // 字节序转换
    uint32_t reverse_bytes(uint32_t value);
    
    // 加载图像数据
    std::vector<Eigen::VectorXd> load_images(const std::string& path, int num_samples);
    
    // 加载标签数据
    std::vector<Eigen::VectorXd> load_labels(const std::string& path, int num_samples);
    
public:
    MNISTLoader(const std::string& train_img_path,
                const std::string& train_lbl_path,
                const std::string& test_img_path,
                const std::string& test_lbl_path);
    
    // 加载训练数据
    std::pair<std::vector<Eigen::VectorXd>, std::vector<Eigen::VectorXd>> 
    load_training_data(int num_samples = 60000);
    
    // 加载测试数据
    std::pair<std::vector<Eigen::VectorXd>, std::vector<Eigen::VectorXd>> 
    load_test_data(int num_samples = 10000);
};

#endif