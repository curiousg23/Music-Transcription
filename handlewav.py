import sys
import wave
import scipy.signal as sig
from pydub import AudioSegment
import numpy as np
import audioop as aio
from cqt import CQT

def preprocess(filename):
    # takes in a filename (.wav format usually), and converts it from audio
    # to a data string. (this may not be needed?) Assuming a sample rate of
    # 44.1 kHz, preprocess the audio into a 252 x n matrix, where n is the
    # number of frames using the CQT transform with 36 bins per octave, and with
    # a hopsize of 512

    #read in audio file, convert to 16-bit int
    audiofile = AudioSegment.from_file(filename)
    data = np.fromstring(audiofile._data, np.int16)

    # downsample to 16kHz
    data = sig.decimate(data, 88200/16000)
    data = data.reshape(data.size,1)

    # assume 44.1 kHz, might downsample later
    sample_rate = 44100
    bins = 36
    # should min, max frequency be limited by piano-produced ranges?
    fmax = 4186.01
    fmin = 27.5
    cqt_audio = CQT(fmin, fmax, bins, sample_rate)
    cqt_audio.gen_cqt_kernel()
    cqt_audio.calc_CQT(data)
    cqt_audio.conv_sparse()

    print cqt_audio.sparse_cqt_coeff.shape

    return cqt_audio.sparse_cqt_coeff

preprocess('test.wav')
