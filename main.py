import itertools
from os import listdir
from os.path import join
from keras.models import Model
from keras.layers import Input, Dense
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import unet
import random
import numpy as np
import keras


class DatasetGenerator:
    def __init__(self):
        paths = list(itertools.chain.from_iterable(
            (map(lambda e: join("dataset", str(i), e),
                 listdir(join("dataset", str(i))))
                for i in range(5))
        ))
        random.shuffle(paths)

        self.generator = DatasetGenerator.__generator(paths)

    @staticmethod
    def __generator(data):
        while data:
            image = Image.open(data[-1])
            image = image.convert('1')
            x_train = list(map(lambda e: e / 255, image.getdata()))

            index = int(data[-1].split("/")[1])
            y_train = [0] * 5
            y_train[index] = 1
            yield (np.array([np.array(x_train)]), 
                   np.array([np.array(y_train)]))
            data.pop()


if __name__ == "__main__":
    datasetGenerator = DatasetGenerator()
    data = datasetGenerator.generator

    inp = Input(shape=(280 * 280,))
    hidden_1 = Dense(512, activation='relu')(inp)
    hidden_2 = Dense(512, activation='relu')(hidden_1)
    out = Dense(5, activation='softmax')(hidden_2)

    model = Model(inputs=inp, outputs=out)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit_generator(data, verbose=1, steps_per_epoch=10, epochs=40)

    # x_test = datasetGenerator.data_generator
    # y_test = datasetGenerator.answer_generator
    model.evaluate_generator(data, 10)

    # model = keras.applications.resnet50.ResNet50(input_shape=(224, 224, 3))
    # model.fit_generator(x_train, y_train)
