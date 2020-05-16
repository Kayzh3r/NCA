from tensorflow import keras
from pydub import AudioSegment, effects


class GRU:
    def __init__(self, checkpoint=None):
        if checkpoint:
            self.__model = keras.models.load_model(checkpoint)
        else:
            self.__model = keras.models.Sequential()

    def __save(self, filename):
        self.__model.save(filename)

    def __loadModel(self, path):
        return

    def train(self, xTrain, yTrain):
        pass
        