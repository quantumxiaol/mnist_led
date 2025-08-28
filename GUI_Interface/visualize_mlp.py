import sys
from PyQt5.QtWidgets import QApplication
from pythoncode.network.MLPNetwork import MLPNetworkG
from pythoncode.visual.neural_visualizer import create_visualizer

if __name__ == '__main__':
    app, window = create_visualizer(
        model_class=MLPNetworkG,
        model_weights_path='./saved_weights/mlp_nn_deeper_mnist.pth',
        window_title='MLP Neural Network 可视化'
    )
    window.show()
    sys.exit(app.exec_())