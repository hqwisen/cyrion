import matplotlib.pyplot as plt
import numpy as np

from keras import Sequential
from keras.datasets import mnist
from keras.layers import Dense
from keras.utils import np_utils

NUMBER_OF_DIGITS = 10


class Datasets:

    @staticmethod
    def reshape_rect(arr):
        """
        Reshape array of square matrix, to array of 1-dim.
        The array element must be 2-dim (rectangle) matrix, and will
        be transformed into 1-dim array of element.width * element.height.
        :param arr: Array of matrix to transform
        :return: Reshaped array with 1-dim elements.
        """
        # arr.shape is (number_of_element, element height, element width)
        return arr.reshape(arr.shape[0], arr.shape[1] * arr.shape[2])

    def __init__(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()
        self.nb_class = NUMBER_OF_DIGITS
        self.img_width, self.img_height = 28, 28
        self.nb_pixels = self.img_width * self.img_height
        self.pre_processing()

    def normalized(self):
        self.x_train = self.x_train.astype('float32')
        self.x_test = self.x_test.astype('float32')
        self.x_train /= 255
        self.x_test /= 255

    def pre_processing(self):
        """
        Transform x_train and x_test, and categorize them.
        Image repr. by h*w  1-dim array.
        Each image is normalized.
        Labels are binary representation of the class.
        """
        self.x_train = Datasets.reshape_rect(self.x_train)
        self.x_test = Datasets.reshape_rect(self.x_test)
        self.normalized()
        # Class vector (int) to binary representation
        self.y_train = np_utils.to_categorical(self.y_train, self.nb_class)
        self.y_test = np_utils.to_categorical(self.y_test, self.nb_class)

    def display(self, nb_sample):
        x_data = self.x_train
        x_data = x_data.reshape(x_data.shape[0], 1, 28, 28)
        sample = x_data[nb_sample]
        print(sample[0])
        plt.imshow(sample[0], cmap='binary')
        plt.savefig('testimg', dpi=1)
        # plt.show()

class NeuralNetwork:

    def __init__(self, dataset, epochs=10, batch_size=200):
        self.dataset = dataset
        self.epochs = epochs
        self.batch_size = batch_size
        self.scores = None
        self.model = self.baseline_model()

    def baseline_model(self):
        model = Sequential()
        model.add(Dense(self.dataset.nb_pixels,
                        input_dim=self.dataset.nb_pixels,
                        kernel_initializer='normal',
                        activation='relu'))
        model.add(Dense(self.dataset.nb_class,
                        kernel_initializer='normal',
                        activation='softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam', metrics=['accuracy'])
        return model

    def train(self):
        validation = (self.dataset.x_test, self.dataset.y_test)
        self.model.fit(self.dataset.x_train, self.dataset.y_train, validation_data=validation,
                       epochs=self.epochs, batch_size=self.batch_size, verbose=2)

    def evaluate(self):
        self.scores = self.model(self.dataset.x_test, self.dataset.y_test, verbose=2)


def load_img(path):
    import glob
    from scipy import misc
    data = misc.imresize(misc.imread(path), (28, 28))
    data = np.asarray(data)
    print(data)
    # data.reshape(1, 1, 28, 28)
    # print(data)
    # x_data = self.x_train
    # x_data = x_data.reshape(x_data.shape[0], 1, 28, 28)
    # sample = x_data[nb_sample]
    # plt.imshow(data, cmap='binary')
    # plt.show()
    # plt.savefig('testimg')




def main():
    # load_img('testimg.png')
    dataset = Datasets()
    dataset.display(1)
    # nn = NeuralNetwork(dataset)
    # np.set_printoptions(threshold=np.inf)
    # print(dataset.y_train)
    # nn.train()
    # x = dataset.x_train.reshape(dataset.x_train.shape[0], dataset.nb_pixels)
    # nn.evaluate()
    # print("Baseline Error: %.2f%%" % (100 - nn.scores[1] * 100))


if __name__ == "__main__":
    main()
