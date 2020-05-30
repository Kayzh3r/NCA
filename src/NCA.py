import argparse
import os
import re
import tempfile
import logging
import numpy as np
from time import time
from pydub import AudioSegment, effects

from src.DBManager import DBManager
from src.NoiseManager import NoiseManager
from src.AudioBooksManager import AudioBooksManager
from src.errors import InitializationError, ResamplingError


class NCA:
    def __init__(self, modelName, modelVersion, modelPyFile=None):
        self.__chromeDriverPath = r"resources/chromedriver"
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

    @staticmethod
    def __importClass(py_file_path, class_name):
        module = '.'.join(list(filter(None, re.split(r'\\+|/|\./|\.\\+', os.path.splitext(py_file_path)[0]))))
        mod = __import__(module, fromlist=['object'])
        mod = getattr(mod, class_name)
        return mod

    def initialize(self):
        try:
            if not os.path.exists(self.__checkpoint_folder):
                logging.info('Checkpoint path ' + self.__checkpoint_folder + ' does not exist. Create folder')
                os.mkdir(self.__checkpoint_folder)
            logging.info('Loading model ' + self.__modelName + self.__modelVer)
            self.__modelLoad()
        except Exception as e:
            logging.error(str(e), exc_info=True)
            raise

    def __modelSave(self, checkpointPath):
        self.__model.save(checkpointPath)

    def __modelLoad(self):
        try:
            modelInfo = self.__db.modelGetInfo(self.__modelName, self.__modelVer)
            if modelInfo is None:
                logging.info('Requested model does not exist in data base')
                if not self.__modelPyFile:
                    raise InitializationError('Model does not exists and path is not given.\n' +
                                              '\rYou must create model before')
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
        except Exception as e:
            logging.error(str(e), exc_info=True)
            raise

    def __checkpointGetUniqueFileName(self):
        checkpointPath = os.path.join(self.__checkpoint_folder,
                                      self.__modelName +
                                      self.__modelVer + '_' +
                                      next(tempfile._get_candidate_names()) + '.h5')
        return checkpointPath

    def train(self, train_time=0, download=0):
        try:
            training = True
            if download:
                logging.info('Downloading audio books for training model')
                self.__audioBooks.downloadData()
                logging.info('Downloading noise audios for training model')
                self.__noise.downloadData()
            logging.info('Start counting training time')
            start_time = time()
            stop_old = start_time
            while training:
                try:
                    logging.info('Retrieving next train combination')
                    nextTrain = self.__db.modelTrainNext(self.__modelName, self.__modelVer)
                    if not nextTrain:
                        logging.info('No combination retrieved. Creating new epoch training combination for model ' +
                                     self.__modelName + self.__modelVer)
                        self.__db.modelTrainNewEpoch(self.__modelName, self.__modelVer,
                                                     self.__model.input_sampling_rate)
                        nextTrain = self.__db.modelTrainNext(self.__modelName, self.__modelVer)
                    training_id = nextTrain[0][0]
                    trackId = nextTrain[0][5]
                    track = self.__db.audioBookGetById(trackId)
                    trackSound = self.load_audio(track[0][9], normalized=False)
                    noiseId = nextTrain[0][6]
                    noise = self.__db.noiseGetById(noiseId)
                    noiseSound = self.load_audio(noise[0][3], normalized=False)
                    combinedSounds = trackSound.overlay(noiseSound)
                    track_samples = np.array(trackSound.get_array_of_samples(), dtype=np.float32)
                    combined_samples = np.array(combinedSounds.get_array_of_samples(), dtype=np.float32)
                    self.__model.train(combined_samples, int(combinedSounds.frame_rate),
                                       track_samples, int(trackSound.frame_rate))
                    self.__db.modelTrainUpdateStatus(training_id, 'TRAINED')
                    stop_new = time()
                    logging.info('Time expended in this training iteration %.3f' % (stop_new - stop_old))
                    stop_old = stop_new
                    total_time = (stop_old - start_time)/60
                    if total_time > train_time:
                        logging.info('Time expended training %.3f exceeds the time requested %.3f'
                                     % (total_time, train_time))
                        checkpointPath = self.__checkpointGetUniqueFileName()
                        logging.info('Saving model checkpoint ' + checkpointPath)
                        self.__modelSave(checkpointPath)
                        logging.info('Registering model ' + self.__modelName + self.__modelVer +
                                     ' checkpoint in data base')
                        self.__db.modelCreateCheckpoint(self.__modelName, self.__modelVer, checkpointPath, "TRAINING")
                        training = False

                except ResamplingError as e:
                    logging.warning(str(e), exc_info=True)
                    self.__db.modelTrainUpdateStatus(training_id, 'RESAMPLING ERROR')

        except Exception as e:
            logging.error(str(e), exc_info=True)
            raise

    def predict(self, xPredict):
        return 0

    @staticmethod
    def load_audio(path, normalized=True):
        ext = os.path.splitext(path)[1][1:]
        logging.info('Loading noise ' + path + ' with file type ' + ext)
        rawSound = AudioSegment.from_file(path, ext)
        if rawSound.channels != 1:
            logging.info('Audio contains more than one channel. Setting to single channel')
            rawSound = rawSound.set_channels(1)
        if normalized:
            logging.info('Normalize noise')
            return effects.normalize(rawSound)
        else:
            return rawSound


if __name__ == "__main__":
    try:
        # set up logging to file
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s %(name)-20s %(levelname)-8s %(message)s',
                            datefmt='%m-%d %H:%M',
                            filename='./NCA.log',
                            filemode='a+')
        # define a Handler which writes DEBUG messages or higher to the sys.stderr
        console = logging.StreamHandler()
        console.setLevel(logging.DEBUG)
        # set a format which is simpler for console use
        formatter = logging.Formatter('%(asctime)s %(name)-20s %(levelname)-8s %(message)s')
        # tell the handler to use this format
        console.setFormatter(formatter)
        # add the handler to the root logger
        logging.getLogger('').addHandler(console)

        logging.info('Starting program execution')
        parser = argparse.ArgumentParser()
        parser.add_argument("model", help="model name for working with", type=str)
        parser.add_argument("ver", help="version of the model", type=str)
        parser.add_argument("-t", "--train", help="train time in minutes", type=int)
        parser.add_argument("-p", "--predict", help="predict input file", type=str)
        parser.add_argument("-c", "--create", help="create new model from python file", type=str)
        parser.add_argument("-d", "--download", action='count', help="Download data and log into database", default=0)
        args = parser.parse_args()
        nca = NCA(args.model, args.ver, args.create)
        if args.train:
            nca.train(args.train, args.download)
        if args.predict:
            trackSound = NCA.load_audio(
                'downloads/don_quijote_2_1001_librivox_64kb_mp3/quijote_vol2_00_cervantes_64kb.mp3', normalized=False)
            noiseSound = NCA.load_audio(
                'downloads/city4', normalized=False)
            combinedSounds = trackSound.overlay(noiseSound)
            track_samples = np.array(trackSound.get_array_of_samples(), dtype=np.float32)
            combined_samples = np.array(combinedSounds.get_array_of_samples(), dtype=np.float32)
            #output = nca.predict(args.predict)
            output = nca.__model.predict(combined_samples, combinedSounds.frame_rate)
    except Exception as e:
        logging.error('Something was wrong', exc_info=True)
