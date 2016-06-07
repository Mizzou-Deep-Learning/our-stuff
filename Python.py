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
filename = 'training_mood.csv'#askopenfilename()
Ltime = 15.0
Utime = 20.0
fields =['period', 'time','HR']
df = pd.read_csv(filename, usecols=fields)
ddf = df.dropna()
print ddf
#plt.show( ddf.plot(kind = 'scatter', x = 'time' , y = 'HR', c = 'c'))
ddf = ddf[ddf.period == '10/3/2014']
ddf = ddf.drop_duplicates()

print Ltime
print "a"

dddf = ddf
dddf = dddf[dddf.time > Ltime]
#dddf = dddf.drop(dddf[dddf.time>Ltime].index)#
#dddf = dddf.drop(dddf[dddf.time > Ltime].index)
#dddf = dddf.drop(dddf[dddf.time < Utime].index)
dddf = dddf[dddf.time < Utime]


print len(dddf['period'])
print len(ddf['period'])
print dddf
heatmap, xedges, yedges = np.histogram2d(dddf['time'], dddf['HR'], bins =(150,30))
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

#print (ddf)

plt.clf()
plt.imshow(heatmap.T,extent=extent,aspect = 'auto',interpolation='nearest')
plt.colorbar()
plt.show()
