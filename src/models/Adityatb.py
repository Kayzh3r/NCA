import logging

import numpy as np
from scipy.signal import decimate
from scipy.signal import spectrogram
from scipy.signal import istft
from scipy.signal import get_window
from librosa.core import amplitude_to_db
from tensorflow import keras
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


class DataGenerator(keras.utils.Sequence):
    """Generates data for Keras"""
    def __init__(self, h5_file_path, batch_size=32, noise=[], shuffle=True):
        self.batch_size = batch_size
        self.h5_file_path = h5_file_path
        self.noise = noise
        self.shuffle = shuffle
        self.h5_info = None

        self.__startup()
        self.on_epoch_end()

    def __startup(self):
        with H5File(self.h5_file_path) as f:


    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        """Generates data containing batch_size samples"""  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = np.load('data/' + ID + '.npy')

            # Store class
            y[i] = self.labels[ID]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)


class Adityatb:
    def __init__(self, checkpoint=None):
        self.batch_size = 32
        self.reg = 0.05
        self.learning_rate = 1e-5
        self.n_units = 600
        self.decay = 1e-3
        self.input_sampling_rate = 11025
        self.n_samples_window = 1024
        self.n_samples_spectrum = int(1024/2) + 1
        self.overlap = 0.5
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

        opt = Adam(lr=self.learning_rate, decay=self.decay)
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

    def train(self, dirtyAudio, dirty_sampling_rate):
        self.__model.fit(db_values_dirty, db_values_clean,
                         batch_size=self.batch_size,
                         epochs=20)

    def predict(self, dirtyAudio, sampling_rate):
        _, _, db_values_dirty, phase = self.__prepateInput(dirtyAudio, sampling_rate)
        clean_mod = self.__model.predict(db_values_dirty)
        clean_audio = self.__prepareOutput(clean_mod,phase)
        return clean_audio
