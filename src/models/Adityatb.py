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
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

logger = logging.getLogger('Adityatb Model')


def compute_soft_mask(y1, y2):
    y1 = np.abs(y1)
    y2 = np.abs(y2)
    m1 = np.divide(y1, np.add(y1, y2))
    m2 = np.divide(y2, np.add(y1, y2))
    # m2 = 1 - m1
    return [m1, m2]


def soft_masking(y):
    input = y[0]
    y1_hat = y[1]
    y2_hat = y[2]
    s1, s2 = compute_soft_mask(y1_hat,y2_hat)
    y1_tilde = np.multiply(s1, input)
    y2_tilde = np.multiply(s2, input)
    return [y1_tilde, y2_tilde]


def masked_out_shape(shape):
    shape_0 = list(shape[0])
    shape_1 = list(shape[1])
    return [tuple(shape_0),tuple(shape_1)]


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
            self.__model = keras.models.load_model(checkpoint)
        else:
            logger.info('Creating new model')
            self.__createModel()

    def __createModel(self):
        regularizer = l2(self.reg)
        input_layer = Input(shape=[1,self.n_samples_spectrum])
        hid1 = LSTM(self.n_units, return_sequences=True, activation='relu')(input_layer)
        dp1 = Dropout(0.2)(hid1)
        hid2 = LSTM(self.n_units, return_sequences=True, activation='relu')(dp1)
        dp2 = Dropout(0.2)(hid2)
        hid3 = LSTM(self.n_units, return_sequences=True, activation='relu')(dp2)
        y1_hat = TimeDistributed(Dense(self.n_samples_spectrum, activation='softmax',
                                       input_shape=[1,self.n_samples_spectrum]),
                                       name='y1_hat')(hid3)
        out_layer = Dense(self.n_samples_spectrum, activation='softmax',
                           name='softMask')(y1_hat)
        #out_layer = Lambda(soft_masking, output_shape=masked_out_shape,
        #                   name='softMask')([input_layer, y1_hat])

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
        factor = input_sampling_rate/self.input_sampling_rate
        logger.info('Input sampling rate is different from the expected by the model.\n' +
                    '\rInput sampling rate: ' + str(input_sampling_rate) + '\n' +
                    '\rModel sampling rate: ' + str(self.input_sampling_rate) + '\n' +
                    'Resampling input signal by factor: ' + str(factor))
        in_signal = decimate(input_signal, factor)
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

    def train(self, dirtyAudio, cleanAudio, sampling_rate):
        _, _, db_values_dirty, _ = self.__prepateInput(dirtyAudio, sampling_rate)
        _, _, db_values_clean, _ = self.__prepateInput(cleanAudio, sampling_rate)
        self.__model.fit(db_values_dirty, db_values_dirty,
                         batch_size=self.batch_size,
                         epochs=1,
                         validation_split=0.1)

    def predict(self, dirtyAudio, sampling_rate):
        _, _, db_values_dirty, phase = self.__prepateInput(dirtyAudio, sampling_rate)
        clean_mod = self.__model.predict(db_values_dirty)
        clean_audio = self.__prepareOutput(clean_mod,phase)
        return clean_audio