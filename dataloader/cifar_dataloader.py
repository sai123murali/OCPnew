# Implement the CIFAR dataset loader

from .base_dataloader import BaseDataLoader
from configs.cifar_config import CIFARConfig
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

class CIFARDataloader(BaseDataLoader):
    def load_data(self):
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        y_train = to_categorical(y_train, CIFARConfig.num_classes)
        y_test = to_categorical(y_test, CIFARConfig.num_classes)
        return (x_train, y_train), (x_test, y_test)
