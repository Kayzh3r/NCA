import argparse
import os
import tempfile

from tensorflow import keras
from pydub import AudioSegment, effects

from src.DBManager import DBManager
from src.NoiseManager import NoiseManager
from src.AudioBooksManager import AudioBooksManager


class NCA:
    def __init__(self, modelName, modelVersion, modelPyFile=None):
        self.__chromeDriverPath = r"C:\Program Files (x86)\Google\ChromeDriver\chromedriver.exe"
        self.db = DBManager()
        self.noise = NoiseManager(self.db)
        self.audioBooks = AudioBooksManager(self.db, self.__chromeDriverPath)
        self.__model = None
        self.__modelName = modelName
        self.__modelVer = modelVersion
        self.__modelPyFile = modelPyFile

        # Calling initialize method
        self.initialize()

    def __importClass(self, name):
        components = name.split('.')
        mod = __import__(components[0])
        for comp in components[1:]:
            mod = getattr(mod, comp)
        return mod

    def initialize(self):
        self.__modelLoad()

    def __modelSave(self, checkpointPath):
        self.__model.save(checkpointPath)

    def __modelLoad(self):
        modelInfo = self.db.modelGetInfo(self.__modelName, self.__modelVer)
        if modelInfo is None:
            self.__modelCreate()
        else:
            self.__model = keras.models.load_model(modelInfo['checkpoint_path'])

    def __modelCreate(self):
        classObj = self.__importClass(self.__modelPyFile)
        self.__model = classObj()
        checkpointPath = self.__checkpointGetUniqueFileName(os.path.dirname(self.__modelPyFile))
        self.__modelSave(checkpointPath)
        self.db.modelCreate(self.__modelName, self.__modelVer, self.__modelPyFile, checkpointPath)
        return 0

    def __checkpointGetUniqueFileName(self, folderPath):
        f = tempfile.NamedTemporaryFile(prefix=self.__modelName + self.__modelVer,
                                        suffix='.h5',
                                        dir=folderPath)
        checkpointPath = f.name
        f.close()
        return checkpointPath

    def train(self, time=0):
        nextTrain = self.db.modelTrainNext(self.__modelName, self.__modelVer)
        if not nextTrain:
            self.db.modelTrainNewEpoch(self.__modelName, self.__modelVer)
        combined_sounds = sound1 + sound2
        rawsound = AudioSegment.from_file("./input.m4a", "m4a")
        normalizedsound = effects.normalize(rawsound)
        normalizedsound.export("./output.wav", format="wav")

    def predict(self):
        return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="model name for working with", type=str)
    parser.add_argument("ver", help="version of the model", type=str)
    parser.add_argument("-t", "--train", help="train time in minutes", type=int)
    parser.add_argument("-p", "--predict", help="predict input file", type=str)
    parser.add_argument("-c", "--create", help="create new model from python file", type=str)
    args = parser.parse_args()
    nca = NCA(args.model, args.ver, args.create)
