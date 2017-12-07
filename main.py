import itertools
from os import listdir
from os.path import join
from keras.models import Sequential
from PIL import Image
import unet
import random
import numpy as np


class DatasetGenerator:
    def __init__(self):
        paths = list(itertools.chain.from_iterable(
            (map(lambda e: join("dataset", str(i), e),
                 listdir(join("dataset", str(i))))
                for i in range(5))
        ))
        random.shuffle(paths)

        self.data_generator = DatasetGenerator.__data_generator(paths)

        self.answer_generator = DatasetGenerator.__answer_generator(paths)

    @staticmethod
    def __data_generator(data):
        while data:
            image = Image.open(data[-1])
            image = image.convert('RGB')
            y_train = list(map(lambda x: np.array(list(map(lambda e: e / 255, x))),
                               image.getdata()))
            yield np.array(y_train)
            data.pop()

    @staticmethod
    def __answer_generator(data):
        while data:
            index = int(data[-1].split("/")[1])
            y_train = [0] * 5
            y_train[index] = 1
            yield np.array(y_train)


if __name__ == "__main__":
    datasetGenerator = DatasetGenerator()
    x_train = datasetGenerator.data_generator
    y_train = datasetGenerator.answer_generator

    model = unet.unet4(3, 280, 280)
