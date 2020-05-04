import argparse
import imp

from tensorflow import keras
from src.DBManager import DBManager
from src.NoiseManager import NoiseManager

class NCA:
    def __init__(self, modelName, modelVersion, modelPyFile=None):
        self.db = DBManager()
        self.noise = NoiseManager(self.db)
        self.__model = None
        self.__modelName   = modelName
        self.__modelVer    = modelVersion
        self.__modelPyFile = modelPyFile

        # Calling initialize method
        self.initialize()

    def initialize(self):
        self.__modelLoad()

    def __modelSave(self):
        self.model.save()

    def __modelLoad(self):
        modelInfo = self.db.modelGetInfo(self.__modelName, self.__modelVer)
        if modelInfo is None:
            self.__modelCreate()
        else:
            self.__model = keras.models.load_model(modelInfo['checkpoint_path'])

    def __modelCreate(self):
        self.__model = imp.load_source(self.__modelName, self.__modelPyFile)
        return 0

    def train(self):
        return 0

    def predict(self):
        return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="model name for working with", type=str)
    parser.add_argument("ver", help="version of the model", type=str)
    parser.add_argument("-t","--train", help="train time in minutes", type=int)
    parser.add_argument("-p","--predict", help="predict input file", type=str)
    parser.add_argument("-c", "--create", help="create new model from python file", type=str)
    args = parser.parse_args()
    nca  = NCA(args.model, args.ver, args.create)

