import matplotlib.pyplot as plt
import numpy as np
import logging


from keras import Sequential
from keras.models import load_model
from keras.datasets import mnist
from keras.layers import Dense
from keras.utils import np_utils

from PIL import Image


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

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


class NeuralNetworkException(Exception):
    pass

class NeuralNetwork:

    def __init__(self, dataset=None, from_file=None, epochs=10, batch_size=200):
        self.dataset = dataset
        self.model_filename = from_file
        self.epochs = epochs
        self.batch_size = batch_size
        self.scores = None
        self.model = self.baseline_model()

    def baseline_model(self):
        """
        :return:
        """
        if not self.model_filename and not self.dataset:
            raise NeuralNetworkException("No model filename or dataset given,"
                                         " cannot build NN model.")
        if self.model_filename:
            logger.debug("Logging NN model from file: %s" % self.model_filename)
            return load_model(self.model_filename)
        else:
            return self.ocr_model()

    def ocr_model(self):
        logger.debug("Building OCR NN model")
        model = Sequential()
        model.add(Dense(self.dataset.nb_pixels,
                        input_dim=self.dataset.nb_pixels,
                        kernel_initializer='normal',
                        activation='relu'))
        model.add(Dense(self.dataset.nb_class,
                        kernel_initializer='normal',
                        activation='softmax'))
        model.compile(loss='binary_crossentropy',
                      optimizer='adam', metrics=['accuracy'])
        return model

    def train(self):
        logger.debug("Training NeuralNetwork..")
        validation = (self.dataset.x_test, self.dataset.y_test)
        self.model.fit(self.dataset.x_train, self.dataset.y_train, validation_data=validation,
                       epochs=self.epochs, batch_size=self.batch_size, verbose=2)

    def save(self, filename=None):
        if not filename:
            filename = NeuralNetwork.DEFAULT_FILENAME
        logger.debug("Saving NN model to %s" % filename)
        self.model.save(filename)

    def evaluate(self):
        self.scores = self.model(self.dataset.x_test, self.dataset.y_test, verbose=2)


NeuralNetwork.DEFAULT_FILENAME = "basic_ocr.h5"


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


def old_main():
    # load_img('testimg.png')
    # dataset = Datasets()
    # img = np.load('arr.npy')
    # img = img.reshape(28, 28)
    # # img.reshape()
    # x_data = self.x_train
    # x_data = x_data.reshape(x_data.shape[0], 1, 28, 28)
    # sample = x_data[nb_sample]
    # print(sample[0])
    # plt.imshow(img, cmap='binary')
    # plt.savefig('testimg', dpi=1)
    # plt.show()
    # dataset.display(1)
    # print(len(dataset.x_train[0]))
    # nn = NeuralNetwork(dataset)
    # np.set_printoptions(threshold=np.inf)
    # print(dataset.y_train)
    # nn.train()
    # x = dataset.x_train.reshape(dataset.x_train.shape[0], dataset.nb_pixels)
    # nn.evaluate()
    # print("Baseline Error: %.2f%%" % (100 - nn.scores[1] * 100))
    pass


def main():
    # nn = NeuralNetwork(from_file=NeuralNetwork.DEFAULT_FILENAME)
    # model = nn.model
    # sample = np.load('sample.npy')
    # sample = np.array([sample])
    # print(sample.shape)
    # result = model.predict(sample, batch_size=200)
    # print(result)
    # # nn.train()
    # nn.save()
    dataset = Datasets()
    nn = NeuralNetwork(from_file=NeuralNetwork.DEFAULT_FILENAME)
    np.set_printoptions(threshold=np.inf)
    # nn.train()
    sample = np.load('sample.npy')
    # result = nn.model.predict(np.array([sample]), batch_size=200)
    pred_classes = nn.model.predict_classes(np.asarray([sample, sample]))
    print(pred_classes)



if __name__ == "__main__":
    logger.debug("Startin main..")
    main()
