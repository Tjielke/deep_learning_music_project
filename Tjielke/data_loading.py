import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
from scipy import signal


#, data = wavfile.read('E:/archive/musicnet/musicnet/train_data/1727.wav')
samplerate, data = wavfile.read('../Pytorch/Data/musicnet/musicnet/train_data/1727.wav')


type(data)


plt.specgram(data)
plt.show()


#sample_rate, samples = wavfile.read('E:/archive/musicnet/musicnet/train_data/1728.wav')
sample_rate, samples = wavfile.read('Data/musicnet/musicnet/train_data/1727.wav')
frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)

plt.pcolormesh(times, frequencies, spectrogram)
plt.imshow(spectrogram)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()


# In[ ]:




