import logging

import numpy as np
from argparse import ArgumentParser
from scipy.signal import decimate
from scipy.signal import spectrogram
from scipy.signal import istft
from scipy.signal import get_window
from librosa.core import amplitude_to_db
from tensorflow import keras
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from h5py import File as H5File
from src.errors import ResamplingError

logger = logging.getLogger('Adityatb-based Model')


class GroupInfo:
    def __init__(self):
        self.path = ''
        self.clean_shape = None
        self.dirty_shape = None
        self.n_fft = 0
        self.attributes = dict()


class ModelSaver(Callback):
    def __init__(self, N):
        self.N = N
        self.batch = 0
        self.epoch = 0

    def on_batch_end(self, batch, logs={}):
        if self.batch % self.N == 0:
            name = './checkpoints/Adityatb%16d.h5' % self.batch
            logger.info('Saving model %s' % name)
            self.model.save(name)
        self.batch += 1

    def on_epoch_begin(self, epoch, logs=None):
        name = './checkpoints/Adityatb%4d.h5' % self.epoch



class DataGenerator(keras.utils.Sequence):
    """Generates data for Keras"""
    def __init__(self, h5_file_path, batch_size=32, shuffle=True, fft_size=0):
        self.batch_size = batch_size
        self.h5_file_path = h5_file_path
        self.shuffle = shuffle
        self.fft_size = fft_size
        self.h5_info = None
        self.groups = []
        self.n_fft = 0
        self.group_idx = 0
        self.target_group = None
        self.fft_idx = 0

        self.__startup()
        self.on_epoch_end()

    def __startup(self):
        with H5File(self.h5_file_path, 'r') as f:
            for main_group_key in f.keys():
                main_group = f[main_group_key]
                for group_key in main_group.keys():
                    group = main_group[group_key]
                    self.__parse_group_info(group)
        self.target_group = self.groups[0]

    def __parse_group_info(self, h5_group):
        group_info = GroupInfo()
        group_info.path = h5_group.name
        group_info.clean_shape = h5_group['CLEAN/DB'].shape
        group_info.dirty_shape = h5_group['DIRTY/DB'].shape
        for key in h5_group.attrs.keys():
            group_info.attributes[key] = h5_group.attrs[key]
        if group_info.clean_shape == group_info.dirty_shape:
            logger.info('Found %d FFTs in %s' % (group_info.clean_shape[0], group_info.path))
            group_info.n_fft = group_info.clean_shape[0]
            self.n_fft += group_info.n_fft
            self.groups.append(group_info)
        else:
            logger.info('Number of FFTs mismatch\n\tClean %d\n\tDirty %d' %
                        (group_info.clean_shape[0], group_info.dirty_shape[0]))

    def __increment_target_group(self):
        self.group_idx += 1
        self.target_group = self.groups[self.group_idx]
        self.fft_idx = 0

    def __len__(self):
        """Denotes the number of batches per epoch"""
        n_batches = int(np.floor(self.n_fft / self.batch_size))
        #logger.debug('Number of batches %d' % n_batches)
        return n_batches

    def __getitem__(self, index):
        """Generate one batch of data"""

        # Generate data
        X, Y = self.__data_generation()

        return X, Y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        pass

    def __data_generation(self):
        """Generates data containing in batch_size"""
        X = np.empty((self.batch_size, 1, self.fft_size), dtype=np.float64)
        Y = np.empty((self.batch_size, 1, self.fft_size), dtype=np.float64)

        # Generate data
        with H5File(self.h5_file_path, 'r') as f:
            for idx in range(self.batch_size):
                # Store sample
                X[idx, :, :] = f[self.target_group.path + '/DIRTY/DB'][self.fft_idx, :, :]
                Y[idx, :, :] = f[self.target_group.path + '/CLEAN/DB'][self.fft_idx, :, :]
                self.fft_idx += 1
                if self.fft_idx >= self.target_group.n_fft:
                    self.__increment_target_group()

        return X, Y


class Adityatb:
    def __init__(self, checkpoint=None):
        self.batch_size = 32
        self.reg = 0.05
        self.learning_rate = 1e-3
        self.n_units = 300
        self.decay = 1e-3
        self.input_sampling_rate = 11025
        self.n_samples_window = 1024
        self.n_samples_spectrum = int(1024/2) + 1
        self.overlap = 0.5
        self.training_generator = None
        self.saver_callback = None
        if checkpoint:
            logger.info('Model checkpoint input obtained')
            self.__model = keras.models.load_model(checkpoint.replace('\\', '/'))
        else:
            logger.info('Creating new model')
            self.__createModel()

    def __createModel(self):
        regularizer = l2(self.reg)
        input_layer = Input(shape=(None, self.n_samples_spectrum))
        hid1 = LSTM(self.n_units, return_sequences=True, activation='relu')(input_layer)
        dp1 = Dropout(0.2)(hid1)
        hid2 = LSTM(self.n_units, return_sequences=True, activation='relu')(dp1)
        dp2 = Dropout(0.2)(hid2)
        hid3 = LSTM(self.n_units, return_sequences=True, activation='relu')(dp2)
        y1_hat = TimeDistributed(Dense(self.n_samples_spectrum, activation='softmax',
                                       input_shape=(None, self.n_samples_spectrum)),
                                       name='y1_hat')(hid3)
        out_layer = Dense(self.n_samples_spectrum, activation='softmax',
                           name='softMask')(y1_hat)

        self.__model = Model(inputs=input_layer, outputs=out_layer)
        model_info = []
        self.__model.summary(print_fn=lambda x: model_info.append(x))
        model_info = '\n\t'.join(model_info)
        logger.info(model_info)

        opt = Adam(lr=self.learning_rate#,
                   #decay=self.decay
                   )
        self.__model.compile(loss='kullback_leibler_divergence',
                             optimizer=opt,
                             metrics=['acc', 'mse'])

    def save(self, filename):
        logger.info('Save model to file ' + filename)
        self.__model.save(filename)

    def __resample(self, input_signal, input_sampling_rate):
        if input_sampling_rate % self.input_sampling_rate:
            raise ResamplingError('Downsampling factor is not integer number\n'
                                  '\tInput sampling rate: %d\n' % input_sampling_rate +
                                  '\tTarget sampling rate: %d\n' % self.input_sampling_rate)
        factor = input_sampling_rate/self.input_sampling_rate
        logger.info('Input sampling rate is different from the expected by the model.\n' +
                    '\rInput sampling rate: ' + str(input_sampling_rate) + '\n' +
                    '\rModel sampling rate: ' + str(self.input_sampling_rate) + '\n' +
                    'Resampling input signal by factor: ' + str(factor))
        in_signal = decimate(input_signal, int(factor))
        return in_signal

    def __prepateInput(self, input_signal, sampling_rate):
        if sampling_rate != self.input_sampling_rate:
            input_signal = self.__resample(input_signal, sampling_rate)
        freq, time, stft = spectrogram(
            input_signal, fs=self.input_sampling_rate,
            window=get_window('hann', self.n_samples_window),
            # nperseg=None,
            noverlap=int(self.overlap*self.n_samples_window), nfft=self.n_samples_window,
            # detrend='constant',
            return_onesided=True, scaling='spectrum', axis=-1, mode='complex')
        db_values = amplitude_to_db(np.abs(stft))
        db_values = np.transpose(db_values)[:, np.newaxis, :]
        phase = np.angle(stft)
        return [freq, time, db_values, phase]

    def __prepareOutput(self, mod, phase):
        recons_stft = mod * np.exp(1j * phase)
        data_recovered = istft(recons_stft, fs=self.input_sampling_rate,
                               window=get_window('hann', self.n_samples_window),
                               # nperseg=None,
                               noverlap=int(self.overlap*self.n_samples_window),
                               nfft=self.n_samples_window)
        return data_recovered

    def train(self, file_name, epochs, save):
        self.training_generator = DataGenerator(h5_file_path=file_name, batch_size=32, fft_size=self.n_samples_spectrum)
        self.saver_callback = ModelSaver(save)
        self.__model.fit_generator(generator=self.training_generator,
                                   use_multiprocessing=True,
                                   max_queue_size=50,
                                   workers=16, epochs=epochs,
                                   callbacks=[self.saver_callback])

    def predict(self, dirtyAudio, sampling_rate):
        _, _, db_values_dirty, phase = self.__prepateInput(dirtyAudio, sampling_rate)
        clean_mod = self.__model.predict(db_values_dirty)
        clean_audio = self.__prepareOutput(clean_mod,phase)
        return clean_audio


if __name__ == "__main__":
    try:
        # set up logging to file
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s %(name)-20s %(levelname)-8s %(message)s',
                            datefmt='%m-%d %H:%M',
                            filename='./train.log',
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
        parser.add_argument("-t", "--train", help="Path of the H5 training data", default='')
        parser.add_argument("-e", "--epochs", help="Number of epochs for training the model", type=int, default=20)
        parser.add_argument("-s", "--save", help="Number of epochs for saving the model", type=int, default=400000)
        parser.add_argument("-l", "--load", help="Load model from checkpoint", default='')
        args = parser.parse_args()

        logging.info('Starting program execution')
        if args.load:
            model = Adityatb(checkpoint=args.load)
        else:
            model = Adityatb()
        if args.train:
            model.train(file_name=args.train, epochs=args.epochs, save=args.save)

    except Exception as e:
        logging.error('Something was wrong', exc_info=True)
