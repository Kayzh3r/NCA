import argparse
import os
import re
import tempfile
import logging

from src.DBManager import DBManager
from src.NoiseManager import NoiseManager
from src.AudioBooksManager import AudioBooksManager


# set up logging to file
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-16s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename='./NCA.log',
                    filemode='a+')
# define a Handler which writes DEBUG messages or higher to the sys.stderr
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
# set a format which is simpler for console use
formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
# tell the handler to use this format
console.setFormatter(formatter)
# add the handler to the root logger
logging.getLogger('').addHandler(console)


class NCA:
    def __init__(self, modelName, modelVersion, modelPyFile=None):
        self.__chromeDriverPath = r"C:\Program Files (x86)\Google\ChromeDriver\chromedriver.exe"
        self.__checkpoint_folder = r'checkpoint'
        self.__db = DBManager()
        self.__noise = NoiseManager(self.__db)
        self.__audioBooks = AudioBooksManager(self.__db, self.__chromeDriverPath)
        self.__model = None
        self.__modelName = modelName
        self.__modelVer = modelVersion
        self.__modelPyFile = modelPyFile

        # Calling initialize method
        self.initialize()

    def __importClass(self, py_file_path, class_name):
        module = '.'.join(list(filter(None, re.split(r'\\+|/|\./|\.\\+', os.path.splitext(py_file_path)[0]))))
        mod = __import__(module, fromlist=['object'])
        mod = getattr(mod, class_name)
        return mod

    def initialize(self):
        if not os.path.exists(self.__checkpoint_folder):
            logging.info('Checkpoint path ' + self.__checkpoint_folder + ' does not exist. Create folder')
            os.mkdir(self.__checkpoint_folder)
        logging.info('Loading model ' + self.__modelName + self.__modelVer)
        self.__modelLoad()

    def __modelSave(self, checkpointPath):
        self.__model.save(checkpointPath)

    def __modelLoad(self):
        modelInfo = self.__db.modelGetInfo(self.__modelName, self.__modelVer)
        if modelInfo is None:
            logging.info('Requested model does not exist in data base')
            classObj = self.__importClass(self.__modelPyFile, self.__modelName)
            self.__model = classObj()
            checkpointPath = self.__checkpointGetUniqueFileName()
            logging.info('Creating initial model checkpoint ' + checkpointPath)
            self.__modelSave(checkpointPath)
            logging.info('Registering model ' + self.__modelName + self.__modelVer +
                         ' and initial checkpoint in data base')
            self.__db.modelCreate(self.__modelName, self.__modelVer, self.__modelPyFile, checkpointPath)
        else:
            self.__modelPyFile = modelInfo['path']
            logging.info('Requested model exists in data base and it is located in ' + self.__modelPyFile)
            classObj = self.__importClass(self.__modelPyFile, self.__modelName)
            logging.info('Loading checkpoint ' + modelInfo['checkpoint_path'] +
                         'for model ' + self.__modelName + self.__modelVer)
            self.__model = classObj(modelInfo['checkpoint_path'])

    def __checkpointGetUniqueFileName(self):
        checkpointPath = os.path.join(self.__checkpoint_folder,
                                      self.__modelName +
                                      self.__modelVer + '_' +
                                      next(tempfile._get_candidate_names()) + '.h5')
        return checkpointPath

    def train(self, time=0):
        logging.info('Downloading audio books for training model')
        self.__audioBooks.downloadData()
        logging.info('Downloading noise audios for training model')
        self.__noise.downloadData()
        logging.info('Retrieving next train combination')
        nextTrain = self.__db.modelTrainNext(self.__modelName, self.__modelVer)
        if not nextTrain:
            logging.info('No combination retrieved. Creating new epoch training combination for model ' +
                         self.__modelVer + self.__modelVer)
            self.__db.modelTrainNewEpoch(self.__modelName, self.__modelVer)
            nextTrain = self.__db.modelTrainNext(self.__modelName, self.__modelVer)
        trackId = nextTrain[0][5]
        track = self.__db.audioBookGetById(trackId)
        trackSound = self.__audioBooks.loadAudioBook(track[0][9], normalized=True)
        noiseId = nextTrain[0][6]
        noise = self.__db.noiseGetById(noiseId)
        noiseSound = self.__noise.loadNoise(noise[0][3], normalized=True)
        combinedSounds = trackSound + noiseSound

    def predict(self, xPredict):
        return 0


if __name__ == "__main__":
    logging.info('Starting program execution')
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="model name for working with", type=str)
    parser.add_argument("ver", help="version of the model", type=str)
    parser.add_argument("-t", "--train", help="train time in minutes", type=int)
    parser.add_argument("-p", "--predict", help="predict input file", type=str)
    parser.add_argument("-c", "--create", help="create new model from python file", type=str)
    args = parser.parse_args()
    nca = NCA(args.model, args.ver, args.create)
