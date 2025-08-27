import sys
from PyQt5.QtWidgets import QApplication
from pythoncode.network.SimpleNeuralNetwork import SimpleNeuralNetworkG
from pythoncode.visual.neural_visualizer import create_visualizer

if __name__ == '__main__':
    app, window = create_visualizer(
        model_class=SimpleNeuralNetworkG,
        model_weights_path='./saved_weights/simple_nn_mnist.pth',
        window_title='Simple Neural Network 可视化'
    )
    window.show()
    sys.exit(app.exec_())