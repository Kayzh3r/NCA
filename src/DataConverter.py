import logging
import os
import numpy as np
from argparse import ArgumentParser
from scipy.signal import decimate, spectrogram, get_window
from librosa.core import amplitude_to_db
from pydub import AudioSegment, effects
from src.errors import ResamplingError
from src.DBManager import DBManager
from src.AudioBooksManager import AudioBooksManager
from src.NoiseManager import NoiseManager

logger = logging.getLogger('DataConverter')


class DataManager:
    def __init__(self):
        self.__INPUT_SAMPLING_RATE = int(11025)
        self.__N_SAMPLES_WINDOW = int(2048)
        self.__OVERLAP = 0.5
        self.__CHROME_DRIVER_PATH = r"resources/chromedriver"

        self.__db = DBManager()
        self.__audio_manager = AudioBooksManager(self.__db, self.__CHROME_DRIVER_PATH)
        self.__noise_manager = NoiseManager(self.__db)

    def main(self, download=0):
        try:
            if download:
                logging.info('Downloading audio books for training model')
                self.__audio_manager.downloadData()
                logging.info('Downloading noise audios for training model')
                self.__noise_manager.downloadData()
            logging.info('Retrieving audio-noise combinations')
            file_combinations = self.__db.modelTrainGetCombination(self.__INPUT_SAMPLING_RATE)
            for file_combination in file_combinations:
                try:
                    logging.info('Loading data')
                    clean_info = self.__db.audioBookGetById(file_combination[1])
                    clean = self.load_audio(clean_info[0][9], normalized=False)
                    clean_samples = np.array(clean.get_array_of_samples(), dtype=np.float32)
                    clean_sampling_rate = clean.frame_rate
                    noise_info = self.__db.noiseGetById(file_combination[2])
                    noise = self.load_audio(noise_info[0][3], normalized=False)
                    dirty = clean.overlay(noise)
                    dirty_samples = np.array(dirty.get_array_of_samples(), dtype=np.float32)
                    dirty_sampling_rate = dirty.frame_rate
                    logging.info('Processing data')
                    dirty_freq, dirty_time, dirty_db, dirty_phase = self.__prepateInput(dirty_samples,
                                                                                        dirty_sampling_rate)
                    clean_freq, clean_time, clean_db, clean_phase = self.__prepateInput(clean_samples,
                                                                                        clean_sampling_rate)
                    logging.info('Storing data')
                except ResamplingError as e:
                    logging.warning(str(e), exc_info=True)

        except Exception as e:
            logging.error(str(e), exc_info=True)
            raise

    def __resample(self, input_signal, input_sampling_rate):
        if input_sampling_rate % self.__INPUT_SAMPLING_RATE:
            raise ResamplingError('Downsampling factor is not integer number\n'
                                  '\tInput sampling rate: %d\n' % input_sampling_rate +
                                  '\tTarget sampling rate: %d\n' % self.__INPUT_SAMPLING_RATE)
        factor = input_sampling_rate / self.__INPUT_SAMPLING_RATE
        logger.info('Input sampling rate is different from the expected by the model.\n' +
                    '\rInput sampling rate: ' + str(input_sampling_rate) + '\n' +
                    '\rModel sampling rate: ' + str(self.__INPUT_SAMPLING_RATE) + '\n' +
                    'Resampling input signal by factor: ' + str(factor))
        in_signal = decimate(input_signal, int(factor))
        return in_signal

    def __prepateInput(self, input_signal, sampling_rate):
        if sampling_rate != self.__INPUT_SAMPLING_RATE:
            input_signal = self.__resample(input_signal, sampling_rate)
        freq, time, stft = spectrogram(
            input_signal, fs=self.__INPUT_SAMPLING_RATE,
            window=get_window('hann', self.__N_SAMPLES_WINDOW),
            # nperseg=None,
            noverlap=int(self.__OVERLAP*self.__N_SAMPLES_WINDOW), nfft=self.__N_SAMPLES_WINDOW,
            # detrend='constant',
            return_onesided=True, scaling='spectrum', axis=-1, mode='complex')
        db_values = amplitude_to_db(np.abs(stft))
        db_values = np.transpose(db_values)[:, np.newaxis, :]
        phase = np.angle(stft)
        return [freq, time, db_values, phase]

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
                            filename='./DataConverter.log',
                            filemode='w+')
        # define a Handler which writes DEBUG messages or higher to the sys.stderr
        console = logging.StreamHandler()
        console.setLevel(logging.DEBUG)
        # set a format which is simpler for console use
        formatter = logging.Formatter('%(asctime)s %(name)-20s %(levelname)-8s %(message)s')
        # tell the handler to use this format
        console.setFormatter(formatter)
        # add the handler to the root logger
        logging.getLogger('').addHandler(console)

        parser = ArgumentParser()
        parser.add_argument("-d", "--download", action='count', help="Download data and log into database", default=0)
        args = parser.parse_args()

        logging.info('Starting program execution')
        data_manager = DataManager()
        data_manager.main(args.download)
    except Exception as e:
        logging.error('Something was wrong', exc_info=True)