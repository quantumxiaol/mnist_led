#include "mnist_loader.h"
#include <iostream>
#include <stdexcept>

MNISTLoader::MNISTLoader(const std::string& train_img_path,
                        const std::string& train_lbl_path,
                        const std::string& test_img_path,
                        const std::string& test_lbl_path)
    : train_images_path(train_img_path), train_labels_path(train_lbl_path),
      test_images_path(test_img_path), test_labels_path(test_lbl_path) {}

uint32_t MNISTLoader::reverse_bytes(uint32_t value) {
    return ((value & 0x000000FF) << 24) |
           ((value & 0x0000FF00) << 8)  |
           ((value & 0x00FF0000) >> 8)  |
           ((value & 0xFF000000) >> 24);
}

std::vector<Eigen::VectorXd> MNISTLoader::load_images(const std::string& path, int num_samples) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + path);
    }
    
    uint32_t magic_number = 0;
    uint32_t number_of_images = 0;
    uint32_t n_rows = 0;
    uint32_t n_cols = 0;
    
    file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
    magic_number = reverse_bytes(magic_number);
    
    file.read(reinterpret_cast<char*>(&number_of_images), sizeof(number_of_images));
    number_of_images = reverse_bytes(number_of_images);
    
    file.read(reinterpret_cast<char*>(&n_rows), sizeof(n_rows));
    n_rows = reverse_bytes(n_rows);
    
    file.read(reinterpret_cast<char*>(&n_cols), sizeof(n_cols));
    n_cols = reverse_bytes(n_cols);
    
    std::cout << "Loading " << std::min(static_cast<uint32_t>(num_samples), number_of_images) 
              << " images of size " << n_rows << "x" << n_cols << std::endl;
    
    std::vector<Eigen::VectorXd> images;
    int actual_samples = std::min(num_samples, static_cast<int>(number_of_images));
    
    for (int i = 0; i < actual_samples; ++i) {
        Eigen::VectorXd image(28 * 28);
        for (int j = 0; j < 28 * 28; ++j) {
            unsigned char pixel = 0;
            file.read(reinterpret_cast<char*>(&pixel), sizeof(pixel));
            image(j) = static_cast<double>(pixel) / 255.0; // 归一化到[0,1]
        }
        images.push_back(image);
    }
    
    return images;
}

std::vector<Eigen::VectorXd> MNISTLoader::load_labels(const std::string& path, int num_samples) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + path);
    }
    
    uint32_t magic_number = 0;
    uint32_t number_of_labels = 0;
    
    file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
    magic_number = reverse_bytes(magic_number);
    
    file.read(reinterpret_cast<char*>(&number_of_labels), sizeof(number_of_labels));
    number_of_labels = reverse_bytes(number_of_labels);
    
    std::cout << "Loading " << std::min(static_cast<uint32_t>(num_samples), number_of_labels) 
              << " labels" << std::endl;
    
    std::vector<Eigen::VectorXd> labels;
    int actual_samples = std::min(num_samples, static_cast<int>(number_of_labels));
    
    for (int i = 0; i < actual_samples; ++i) {
        unsigned char label = 0;
        file.read(reinterpret_cast<char*>(&label), sizeof(label));
        
        Eigen::VectorXd one_hot(10);
        one_hot.setZero();
        one_hot(label) = 1.0;
        labels.push_back(one_hot);
    }
    
    return labels;
}

std::pair<std::vector<Eigen::VectorXd>, std::vector<Eigen::VectorXd>> 
MNISTLoader::load_training_data(int num_samples) {
    auto images = load_images(train_images_path, num_samples);
    auto labels = load_labels(train_labels_path, num_samples);
    return std::make_pair(images, labels);
}

std::pair<std::vector<Eigen::VectorXd>, std::vector<Eigen::VectorXd>> 
MNISTLoader::load_test_data(int num_samples) {
    auto images = load_images(test_images_path, num_samples);
    auto labels = load_labels(test_labels_path, num_samples);
    return std::make_pair(images, labels);
}