import mnist_loader
import nnpy

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

net = nnpy.Network([784, 100, 10])

net.SGD(training_data, 30, 10, 3.0, test_data=test_data)