#!/bin/bash

# MNIST数据集下载脚本
echo "=================================="
echo "Downloading MNIST Dataset"
echo "=================================="

# 创建目录
mkdir -p data/MNIST/raw

# 进入数据目录
cd data/MNIST/raw


# MNIST数据集URL
BASE_URL="http://yann.lecun.com/exdb/mnist"
FILES=(
    "train-images-idx3-ubyte.gz"
    "train-labels-idx1-ubyte.gz"
    "t10k-images-idx3-ubyte.gz"
    "t10k-labels-idx1-ubyte.gz"
)

echo "Downloading MNIST files..."
echo "--------------------------"

# 下载文件
for file in "${FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo "Downloading $file..."
        wget --no-check-certificate "$BASE_URL/$file"
        if [ $? -eq 0 ]; then
            echo "✓ Downloaded $file"
        else
            echo "✗ Failed to download $file"
            exit 1
        fi
    else
        echo "✓ $file already exists"
    fi
done

echo ""
echo "Extracting files..."
echo "-------------------"

# 解压文件
for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "Extracting $file..."
        gunzip -f "$file"
        if [ $? -eq 0 ]; then
            echo "✓ Extracted ${file%.gz}"
        else
            echo "✗ Failed to extract $file"
        fi
    fi
done

echo ""
echo "MNIST dataset download and extraction completed!"
echo ""
echo "Files downloaded:"
ls -la *.ubyte 2>/dev/null || echo "No .ubyte files found"

echo ""
echo "Dataset is ready in data/MNIST/raw/"
echo "=================================="