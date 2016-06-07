from Tkinter import Tk
from tkFileDialog import askopenfilename
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("tkAgg")
import matplotlib.pyplot as plt
import scipy.fftpack
from scipy import signal


Tk().withdraw()
filename = askopenfilename()
#fields =['period', 'time','HR','BR', 'activity', 'ambulation']
df = pd.read_csv(filename)#, #usecols=fields)
#ddf = df.dropna()
ddf = df


ddf = ddf[ddf.period == '10/4/2014']
#ddf = ddf[ddf.ambulation == 'Stationary']

ddf = ddf[ ddf.time <= 15.5]
ddf = ddf [ddf.time >= 15]

NFFT = 64
Fs = 2

ax = plt.subplot(111)
datalist = ddf.columns.values

Pxx, freqs, bins, im = plt.specgram(ddf['HR'], NFFT=NFFT, pad_to=NFFT, noverlap=63, Fs=Fs)
plt.colorbar()
plt.show()