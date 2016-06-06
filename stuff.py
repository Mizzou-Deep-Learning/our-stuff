from Tkinter import Tk
from tkFileDialog import askopenfilename
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("tkAgg")
import matplotlib.pyplot as plt
import scipy.fftpack



Tk().withdraw()
filename = askopenfilename()

fields =['period', 'time','HR']
df = pd.read_csv(filename, usecols=fields)
ddf = df.dropna()
#plt.show( ddf.plot(kind = 'scatter', x = 'time' , y = 'HR', c = 'c'))
ddf = ddf[ddf.period == '10/3/2014']
heatmap, xedges, yedges = np.histogram2d(ddf['time'], ddf['HR'], bins =(150,42))
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

print (ddf)

plt.clf()
plt.imshow(heatmap,extent=extent, origin = 'lower',aspect = 'auto',interpolation='nearest')
plt.colorbar()
plt.show()


