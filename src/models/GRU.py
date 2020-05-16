import logging

from tensorflow import keras

logger = logging.getLogger('GRU')


class GRU:
    def __init__(self, checkpoint=None):
        if checkpoint:
            logger.info('Model checkpoint input obtained')
            self.__model = keras.models.load_model(checkpoint)
        else:
            logger.info('Creating new model')
            self.__createModel()

    def __createModel(self):
        self.__model = keras.models.Sequential()
        self.__model.

    def __save(self, filename):
        logger.info('Save model to file ' + filename)
        self.__model.save(filename)

    def __loadModel(self, path):
        return

    def train(self, xTrain, yTrain):
        pass
