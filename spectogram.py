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
fields =['period', 'time','HR','BR', 'activity', 'ambulation']
df = pd.read_csv(filename, usecols = fields)
ddf = df


ddf = ddf[ddf.period == '10/4/2014']


NFFT = 50
Fs = 2
x = 0;
y= 0.5;

for i in range(1000):
    tempddf = ddf
    tempddf = tempddf[tempddf.time >= x]
    tempddf = tempddf[tempddf.time <= y]
    Pxx, freqs, bins, im = plt.specgram(tempddf['HR'], NFFT=np.hamming(50), pad_to=NFFT, noverlap=45, Fs=Fs)
    #plt.colorbar()
    plt.savefig('pic'+ str(i) + '.png', bbox_inches='tight', pad_inches = 0)
    #plt.show()
    x = x + 0.083
    y = y + 0.083
    if y == 24:
        break

