import sys
from PyQt5.QtWidgets import QApplication
from pythoncode.network.LeNet import LeNet5G
from pythoncode.visual.neural_visualizer import create_visualizer

if __name__ == '__main__':
    app, window = create_visualizer(
        model_class=LeNet5G,
        model_weights_path='./saved_weights/lenet_mnist.pth',
        window_title='LeNet-5 可视化'
    )
    window.show()
    sys.exit(app.exec_())