######################################################
#   IMPORTATION FOR IMAGE DISPLAY        #############
######################################################
import glob
import pickle
from sys import argv

import numpy as np
import matplotlib

import logging

matplotlib.use('agg')
import matplotlib.pyplot as plt

plt.switch_backend("TkAgg")
import scipy
# let's keep our keras backend tensorflow quiet
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# for testing on CPU
# os.environ['CUDA_VISIBLE_DEVICES'] = ''
######################################################
######################################################


######################################################
#   IMPORTATION FOR IMAGE PREPROCESSING  #############
######################################################
import random
import cv2
import skimage.morphology as morp
from skimage.filters import rank
from sklearn.utils import shuffle
import csv
import os
import tensorflow as tf
######################################################
######################################################


######################################################
#   KERAS IMPORTATION FOR CNN BUILDING   #############
######################################################
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator

# from keras.layers.core import Dense, Dropout, Activation
######################################################
######################################################

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


class DataManager:
    """
    The data processing (preprocessing,
    extraction, etc.) management
    """

    def __init__(self, dataFiles, preprocess=True):
        """
        dataFiles == list of training data, validation data and test data files
        """
        self.dataFiles = dataFiles
        self.signs = []
        ########################
        self.labelsExtraction()
        if preprocess:
            self.Xtraining, self.Ytraining, self.Xvalidation, self.Yvalidation, self.Xtest, self.Ytest = self.imagesExtraction()
            ########################
            # define 3 data generators for data augmentation
            # self.featureStandardization = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
            # self.zcaWhitening = ImageDataGenerator(zca_whitening=True)
            self.randomShifts = ImageDataGenerator(width_shift_range=0.2,
                                                   height_shift_range=0.2)  # shift = 0.2
            # self.nbSampleToAugment = 20000
            # self.someAugmentedImages = []
            # self.someAugmentedLabels = []
            # ########################
            self.dataShuffling()  # data preprocessing step 1
            self.preprocessing()  # data preprocessing step 2,3,4

    # self.dataAugmentation()

    # self.showImages(self.someAugmentedImages, self.someAugmentedLabels, "gray")

    def labelsExtraction(self):
        """
        Extracts labels data from files
        Maps classID to traffic sign names
        """
        with open("./traffic-signs-data/signnames.csv", 'r') as csvfile:
            signnames = csv.reader(csvfile, delimiter=',')
            next(signnames, None)
            for row in signnames:
                self.signs.append(row[1])
            csvfile.close()

    def imagesExtraction(self):
        """
        Extracts images data from files
        """
        with open(self.dataFiles[0], mode='rb') as f:
            training = pickle.load(f)
        with open(self.dataFiles[1], mode='rb') as f:
            validation = pickle.load(f)
        with open(self.dataFiles[2], mode='rb') as f:
            testing = pickle.load(f)
        return (
            training['features'], training['labels'], validation['features'], validation['labels'],
            testing['features'], testing['labels'])

    def dataShuffling(self):
        """
        First step preprocessing :
        shuffle training data
        """
        self.Xtraining, self.Ytraining = shuffle(self.Xtraining, self.Ytraining)

    def toGrayScale(self, image):
        """
        Second step preprocessing :
        from RGB image to grayscale
        = DIMENSIONALITY REDUCTION
        """
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    def dataNormalization(self, image):
        """
        Third step preprocessing :
        Normalization of image :
        normalize inputs from 0-255 to 0.0-1.0
        """
        image = np.divide(image, 255)
        return image

    def histogramEqualization(self, image):
        """
        Fourth step preprocessing :
        Apply local histogram equalization
        to grayscale images
        """
        kernel = morp.disk(30)
        newImage = rank.equalize(image, selem=kernel)
        return newImage

    def dataPreprocessing(self, images):
        """
        Apply the preprocessing steps 2,3,4
        to images; images = training, testing
        or validation list of images
        """
        grayScaledImages = list(map(self.toGrayScale, images))
        equalizedImages = list(map(self.histogramEqualization, grayScaledImages))
        n_training = images.shape
        normalizedImages = np.zeros((n_training[0], n_training[1], n_training[2]))
        for i, image in enumerate(equalizedImages):
            normalizedImages[i] = self.dataNormalization(image)
        normalizedImages = normalizedImages[..., None]
        return normalizedImages

    def preprocessing(self):
        self.Xtraining, self.Xvalidation, self.Xtest = self.dataPreprocessing(self.Xtraining), \
                                                       self.dataPreprocessing(
                                                           self.Xvalidation), self.dataPreprocessing(
            self.Xtest)

    def dataAugmentation(self):
        """
        Double the number of training samples
        by augmenting them through 3 types of
        augmentation
        """
        # fit parameters from data
        # self.featureStandardization.fit(self.Xtraining)
        # self.zcaWhitening.fit(self.Xtraining)
        self.randomShifts.fit(self.Xtraining)
        # retrieve one batch of images from each data generator
        # self.extractBatchGen(self.featureStandardization, self.nbSampleToAugment)
        # self.extractBatchGen(self.zcaWhitening, self.nbSampleToAugment)
        self.extractBatchGen(self.randomShifts, self.nbSampleToAugment)
        ###############################################################
        self.someAugmentedImages = np.array(self.someAugmentedImages)
        self.someAugmentedLabels = np.array(self.someAugmentedLabels)

    def extractBatchGen(self, dataGenerator, nbSample):
        """
        Configure batch size and retrieve one batch of images from data generator
        nbSample defines the batch size
        """
        sampleToShow = 9
        for Xbatch, Ybatch in dataGenerator.flow(self.Xtraining, self.Ytraining,
                                                 batch_size=nbSample):
            for i in range(nbSample):
                self.Xtraining = np.concatenate((self.Xtraining, np.asarray([Xbatch[i]])), axis=0)
                self.Ytraining = np.concatenate((self.Ytraining, np.asarray([Ybatch[i]])), axis=0)
                if (i < sampleToShow):
                    self.someAugmentedImages.append(Xbatch[i].reshape(32, 32))
                    self.someAugmentedLabels.append(Ybatch[i])
            break

    def showImages(self, X, Y, cmap=None):
        """
        Display a list of images
        """
        fig = plt.figure()
        indx = random.randint(0, 1)
        # Use gray scale color map if there is only one channel
        cmap = 'gray' if len(X[indx].shape) == 2 else cmap
        print(X.shape)
        for i in range(9):
            plt.subplot(3, 3, i + 1)
            plt.tight_layout()
            plt.imshow(X[i].reshape(32, 32), cmap=cmap, interpolation='none')
            plt.xlabel(self.signs[Y[i]])
            plt.xticks([])
            plt.yticks([])
        plt.show()

    def displayExample(self):
        print("Displaying examples")
        # affichage normal
        self.showImages(self.Xtraining, self.Ytraining, "gray")

        # affichage après grisage
        # grayScaledImages = list(map(self.toGrayScale, self.Xtraining))
        # self.showImages(grayScaledImages, self.Ytraining, "gray")
        #
        # # Sample images after Local Histogram Equalization
        # equalizedImages = list(map(self.histogramEqualization, grayScaledImages))
        # self.showImages(equalizedImages, self.Ytraining, "gray")
        #
        # # Sample images after normalization
        # n_training = self.Xtraining.shape
        # normalizedImages = np.zeros((n_training[0], n_training[1], n_training[2]))
        # for i, image in enumerate(equalizedImages):
        #     normalizedImages[i] = self.dataNormalization(image)
        # self.showImages(normalizedImages, self.Ytraining, "gray")
        # normalizedImages = normalizedImages[..., None]

    ################################################################################################################################################


################################################################################################################################################
################################################################################################################################################
################################################################################################################################################
################################################################################################################################################

class DeepCNN:
    """
    Implements a predefined
    CNN architecture
    """

    def __init__(self, aDataManager):
        """
        aDataManager == data feeder
        """
        self.myDataManager = aDataManager
        self.fitLabelData()
        self.fitImages()
        self.nbClass = aDataManager.Ytest.shape[1]  # number of classes
        self.myModel = Sequential()  # CNN
        # fix random seed for reproducibility
        self.seed = 7
        np.random.seed(self.seed)

        ###############################################
        #  (OTHER) HYPERPARAMETERS                    #
        ###############################################
        # for model compiling
        self.epochs = 25
        self.lrate = 0.01
        self.decay = (self.lrate / self.epochs)
        self.sgd = SGD(lr=self.lrate, momentum=0.9, decay=self.decay, nesterov=False)
        #											  #
        ###############################################

        # self.modelBuildingDeeper()
        self.modelBuilding()
        self.modelCompiling()
        self.modelTraining()
        self.modelEvaluation()
        self.modelSaving()

    def fitLabelData(self):
        """
        Fits label data for the CNN to
        one hot encode outputs
        """

        self.myDataManager.Ytraining = np_utils.to_categorical(self.myDataManager.Ytraining)
        self.myDataManager.Ytest = np_utils.to_categorical(self.myDataManager.Ytest)
        self.myDataManager.Yvalidation = np_utils.to_categorical(self.myDataManager.Yvalidation)

    def fitImages(self):
        """
        Fits images data for the CNN :
        from 32x32x3 to 3X32x32 OR
        from 32x32x1 to 1X32x32
        """

        self.myDataManager.Xtraining = np.rollaxis(self.myDataManager.Xtraining, 3, 1)
        self.myDataManager.Xtest = np.rollaxis(self.myDataManager.Xtest, 3, 1)
        self.myDataManager.Xvalidation = np.rollaxis(self.myDataManager.Xvalidation, 3, 1)

    def modelBuilding(self):
        """
        Adding layers
        """
        self.myModel.add(
            Conv2D(32, (3, 3), input_shape=(1, 32, 32), padding='same', activation='relu',
                   kernel_constraint=maxnorm(3)))
        self.myModel.add(Dropout(0.2))
        self.myModel.add(
            Conv2D(32, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
        self.myModel.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))
        self.myModel.add(Flatten())
        self.myModel.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
        self.myModel.add(Dropout(0.5))
        self.myModel.add(Dense(self.nbClass, activation='softmax'))

    def modelBuildingDeeper(self):
        """
        Adding layers
        """
        self.myModel.add(
            Conv2D(32, (3, 3), input_shape=(1, 32, 32), activation='relu', padding='same'))
        self.myModel.add(Dropout(0.2))
        self.myModel.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
        self.myModel.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))
        self.myModel.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        self.myModel.add(Dropout(0.2))
        self.myModel.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        self.myModel.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))
        self.myModel.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        self.myModel.add(Dropout(0.2))
        self.myModel.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        self.myModel.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))
        self.myModel.add(Flatten())
        self.myModel.add(Dropout(0.2))
        self.myModel.add(Dense(1024, activation='relu', kernel_constraint=maxnorm(3)))
        self.myModel.add(Dropout(0.2))
        self.myModel.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
        self.myModel.add(Dropout(0.2))
        self.myModel.add(Dense(self.nbClass, activation='softmax'))

    def modelCompiling(self):
        self.myModel.compile(loss='categorical_crossentropy', optimizer=self.sgd,
                             metrics=['accuracy'])
        print(self.myModel.summary())

    def modelTraining(self):
        np.random.seed(self.seed)
        self.myModel.fit(self.myDataManager.Xtraining, self.myDataManager.Ytraining,
                         validation_data=(
                             self.myDataManager.Xvalidation, self.myDataManager.Yvalidation),
                         epochs=self.epochs, batch_size=64)

    def modelEvaluation(self):
        # Final evaluation of the model
        scores = self.myModel.evaluate(self.myDataManager.Xtest, self.myDataManager.Ytest,
                                       verbose=0)
        print("Accuracy: %.2f%%" % (scores[1] * 100))

    def modelSaving(self):
        # saving the model
        saveDir = "resultsTAI/"
        if not os.path.exists(saveDir):
            os.mkdir(saveDir)
        modelFileName = "trafficTAI.h5"
        modelPath = os.path.join(saveDir, modelFileName)
        self.myModel.save(modelPath)
        print('Saved trained model at %s ' % modelPath)


################################################################################################################################################
################################################################################################################################################
################################################################################################################################################
################################################################################################################################################
################################################################################################################################################

class backendAppli:
    """
    Test the model with
    new loaded images (after
    their preprocessing)
    """

    def __init__(self, imagesFilesList, modelFileName, aDataManager):
        """
        imagesFilesList = list of images (filenames)
        """
        self.DataManager = aDataManager
        self.imagesList = []
        self.imagesLoading(imagesFilesList)  # loading the image file
        self.imagesPreprocessing()
        self.loadedModel = load_model(modelFileName)  # loading the model
        self.predictions = self.predict()
        self.displayPrediction()

    def imagesLoading(self, imagesFilesList):
        logger.debug("Loading images: %s" % imagesFilesList)
        for elem in imagesFilesList:
            self.imagesList.append(cv2.imread(elem, 1))

    def imagesPreprocessing(self):
        """
        Images rescaling and preprocessing
        """
        for i in range(len(self.imagesList)):
            # rescaling the image
            self.imagesList[i] = scipy.misc.imresize(self.imagesList[i], (32, 32), interp="bicubic")
        self.imagesList = list(map(self.DataManager.toGrayScale, self.imagesList))
        self.imagesList = list(map(self.DataManager.histogramEqualization, self.imagesList))
        self.imagesList = np.asarray(self.imagesList).astype('float32')

    def predict(self):
        # classifying the images
        predictions = []
        for elem in self.imagesList:
            predictions.append(self.loadedModel.predict(elem.reshape(1, 1, 32, 32)))
        return predictions

    def displayPrediction(self):
        # plot the figures along with the predictions
        fig = plt.figure()
        # Use gray scale color map if there is only one channel
        cmap = 'gray' if len(self.imagesList[0].shape) == 2 else cmap
        for i in range(len(self.imagesList)):
            # Show 4 images a row
            nrows, ncols = (len(self.imagesList) // 4) + 1, 4
            plt.subplot(4, 3, i + 1)
            plt.tight_layout()
            plt.imshow(self.imagesList[i].reshape(32, 32), cmap=cmap, interpolation='none')
            plt.xlabel("{}".format(self.DataManager.signs[np.nonzero(self.predictions[i])[1][0]]))
            plt.xticks([])
            plt.yticks([])
        plt.show()


if __name__ == '__main__':
    dataSet = DataManager(["./traffic-signs-data/train.p", "./traffic-signs-data/valid.p",
                           "./traffic-signs-data/test.p"], preprocess=False)
    # dataSet.displayExample()
    # cnn = DeepCNN(dataSet)
    # graphique et matrice
    # samples = ["panneaux/im1.jpg", "panneaux/im2.jpg", "panneaux/im3.jpg", "panneaux/im4.jpg",
    #            "panneaux/im5.jpg", "panneaux/im6.jpg",
    #            "panneaux/im7.jpg", "panneaux/im8.jpg", "panneaux/im9.jpg"]
    # samples = ['datasets/70sample.jpg', 'datasets/70sample2.png']
    samples = glob.glob("samples/signs_samples/*.jpg")
    modelfile = "resultsTAI/trafficTAI_deep.h5"
    if len(argv) > 1:
        modelfile = argv[1]
    test = backendAppli(samples, modelfile, dataSet)
