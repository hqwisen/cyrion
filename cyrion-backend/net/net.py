from keras import Sequential
from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.layers import Dense

NUMBER_OF_DIGITS = 10


class Datasets:
    def __init__(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()
        self.nb_class = NUMBER_OF_DIGITS
        self.img_width, self.img_height = 28, 28
        self.nb_pixels = self.img_width * self.img_height

    def display(self, nb_sample):
        x_data = self.x_train
        # print(x_data.shape)
        # x_data = x_data.reshape(x_data.shape[0], 1, 28, 28)
        # print("###############")
        # print(x_data[1])
        # sample = x_data[nb_sample]
        # plt.imshow(sample[0], cmap='binary')
        # plt.show()


class NeuralNetwork:

    def __init__(self, dataset):
        self.dataset = dataset
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


def main():
    dataset = Datasets()
    cnn = NeuralNetwork(dataset)


if __name__ == "__main__":
    main()
