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
df = pd.read_csv(filename)
#ddf = df.dropna()
#print ddf
#plt.show( ddf.plot(kind = 'scatter', x = 'time' , y = 'HR', c = 'c'))
#ddf = ddf[ddf.period == '10/3/2014']
#ddf = ddf.drop_duplicates()


heatmap, xedges, yedges = np.histogram2d(df['time'], df['HR'], bins =(150,30))
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

#print (ddf)

plt.clf()
plt.imshow(heatmap.T,extent=extent,aspect = 'auto',interpolation='nearest')
plt.colorbar()
plt.show()
