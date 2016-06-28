from Tkinter import Tk
from tkFileDialog import askopenfilename
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("tkAgg")
import matplotlib.pyplot as plt
import scipy.fftpack
from scipy import signal
def setNANtoMean(df):
    ddf = df.fillna(df.mean())
    return ddf

TIMECOL = 1
DATECOL = 0
DRINKCOL = 7 #Column to determine whether or not drinking. #TODO: make more configurable
SKIP = [DATECOL, TIMECOL, DRINKCOL, 6,8,9,10,11,12,13,14] #used for determining garbage columns#TODO: Find a workaround to this for more configurability
SELECTEDDATE = '10/3/2014' #target date. #TODO: Should be a range.
Tk().withdraw()
filename = 'training_mood.csv'#askopenfilename() # choose file with this function
Utime = .05 #upper bound for time collection #TODO: More configurable
Ltime = 0 #Lower bound for time collection
window = .5 #for iterating through time

jump = (1.0/12.0) # every 5 minutes
df = pd.read_csv(filename) #turn excel/csv file into a dataframe (large pandas table)


ddf = setNANtoMean(df) # removes all NaN values with the mean of the column

NFFT = 64      # the length of the windowing segments TODO: Figure out what this means/does
Fs = 0.2 #  # the sampling frequency       ### Y axis (seems to alter the x values tho)

imgNum = 0 #iteration var

headers = list(ddf.columns.values)
ddf = ddf[ddf[headers[DATECOL]] == SELECTEDDATE]  #selected date. #########TODO: iterate through dates, as well

namingVar = 0 #used for naming image fil


for i in headers:
      #nfft is hamming window, 50
    namingVar = 0
    if ddf.columns.get_loc(i) not in SKIP:
        print i + "is not in skip"

        for imgNum in np.arange(Ltime, Utime,jump):  #for each image from lower bound to upper bound by step size of step
            plt.cla()  #remove axes
            plt.clf()  #clear plot to be empty
            tempddf = ddf  #temporary dataframe for snapshot in time
            print imgNum  #upper bound for time
            print imgNum + window  #lower bound for time
            tempddf = tempddf[tempddf[headers[TIMECOL]] > imgNum]  #remove lower bound of time
            tempddf = tempddf[tempddf[headers[TIMECOL]] < imgNum + window]  #remove upper bound of time
            #print "LEN OF tempddf: " + str(len(tempddf['period']))  #size of removed data
            if len(tempddf) >= 0:
                ax = plt.subplot(111) #length,width,height ratio of 1:1:1
                Pxx, freqs, bins, im = plt.specgram((tempddf[i]), NFFT=NFFT, noverlap=63, Fs=Fs) #generate spectrogram
                plt.savefig(str(i) + str(namingVar) + '.png', bbox_inches='tight', pad_inches = 0) #save plot
                plt.colorbar()
                namingVar = namingVar + 1
