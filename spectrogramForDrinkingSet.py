from Tkinter import Tk
from tkFileDialog import askopenfilename
import pandas as pd
import numpy as np
import matplotlib


from subprocess import call

matplotlib.use("tkAgg")
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import math
import random
import os
import exceptions
import scipy.fftpack
from scipy import signal
#def setNANtoMean(df):
    #ddf = df.fillna(df.mean())
    #return ddf
def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise

HOUR = 1.0
MINUTE = HOUR/60.0
SECOND = MINUTE/60.0


call(["ls", "-l"])

TIMECOL = 1
DATECOL = 0
DRINKCOL = 7 #Column to determine whether or not drinking. #TODO: make more configurable
SKIP = [DATECOL, TIMECOL, DRINKCOL, 6,8,9,10,11,12,13,14] #used for determining garbage columns#TODO: Find a workaround to this for more configurability
SELECTEDDATE = ['10/3/2014'] #target date. #TODO: Should be a range.
Tk().withdraw()
filename = 'training_mood.csv'#askopenfilename() # choose file with this function
Utime = .05 #upper bound for time collection #TODO: More configurable
Ltime = 0 #Lower bound for time collection
window = .5 #for iterating through time

jump = (1.0/12.0) # every 5 minutes


df = pd.read_csv(filename) #turn excel/csv file into a dataframe (large pandas table)
headers = list(df.columns.values)

dfDrinkPts = df[df[headers[DRINKCOL]] == 1]
#print dfDrinkPts


#tddf = setNANtoMean(df) # removes all NaN values with the mean of the column
#matlab spectrogram
NFFT = 64 #int(pow(2, math.log(1500,2)))#64 #TODO See matlab for formula      # the length of the windowing segments TODO: Figure out what this means/does
Fs = 0.2 # should be samping rate #0.2 #  # the sampling frequency       ### Y axis (seems to alter the x values tho)
NOVERLAP = 49  #63
WINDOW = 50
#window = 50
#do not output observation, output time

#also zoom in on low frequency

imgNum = 0 #iteration var



namingVar = 0 #used for naming image fil



for i, row in dfDrinkPts.iterrows():
    #print str(i) + " - " + str(row)
    tempdf = df[df[headers[DATECOL]] == row["period"]]
    #print tempdf
    tempdf = tempdf[tempdf[headers[TIMECOL]] <= (row["time"] + 15*MINUTE)]
    tempdf = tempdf[tempdf[headers[TIMECOL]] >= (row["time"] - 15*MINUTE)]
    #print str(len(tempdf)) + " : LENGTH"
    #print tempdf
    for j in headers:
        if(tempdf.columns.get_loc(j) not in SKIP):
            print len(tempdf)
            #NFFT = int(pow(2, math.log(len(tempdf),2)))
            #NFFT = len(tempdf)
            #NFFT = 50
            NFFT = 256

            #window=mlab.window_hanning(tempdf[j]),
            #
            Pxx, freqs, bins, im = plt.specgram((tempdf[j]), NFFT=NFFT, noverlap = 49,    Fs=Fs, mode='magnitude')   #, window=mlab.window_hanning(tempdf[j])) #generate spectrogram
            make_sure_path_exists(str(j) + "/drink")
            plt.savefig(str(j) + "/drink/" + str(namingVar/4) + '.png', bbox_inches='tight', pad_inches = 0) #save plot
            #plt.colorbar()
            namingVar = namingVar + 1
                                    #intro, OKR, Process, specgram, deeplearning
numCategories = namingVar/4

spectrogramMade = 0

while spectrogramMade != numCategories:
    time = random.randrange(600, 23400)
    time = float(time)
    time = time/100.0
    #print [(df.columns.values[DATECOL])]
    randDate = df.drop_duplicates(subset=[df.columns.values[DATECOL]])
    #print randDate
    #print len(randDate)
    choose = random.randrange(0, len(randDate))
    #print choose
    choice = df.sample(n=1)
    #print choice
    #print choice.iloc[0,0]
    date = choice.iloc[0,0]
    NoDrinkDf = df[df[headers[DATECOL]] == date]
    #print NoDrinkDf
    #print str(NoDrinkDf[headers[TIMECOL]])
    #print  str(time + 30*MINUTE) + "TIME"
    NoDrinkDf = NoDrinkDf[NoDrinkDf[headers[TIMECOL]] <= (time + 15*MINUTE)]
    NoDrinkDf = NoDrinkDf[NoDrinkDf[headers[TIMECOL]] >= (time - 15*MINUTE)]
    #print NoDrinkDf
    testfordrinkDF = NoDrinkDf[NoDrinkDf[headers[DRINKCOL]] == 1]
    print testfordrinkDF
    print len(testfordrinkDF)#.count()

    print "PRUNTED"
    if len(testfordrinkDF) == 0:

        for j in headers:
            if (tempdf.columns.get_loc(j) not in SKIP):
                ax = plt.subplot(111)  # length,width,height ratio of 1:1:1
                Pxx, freqs, bins, im = plt.specgram((tempdf[j]), NFFT=NFFT, noverlap=NOVERLAP, #window=mlab.window_hanning(tempdf[j]),
                                                Fs=Fs)  # generate spectrogram
                make_sure_path_exists(str(j) + "/no-drink")
                plt.savefig(str(j) + "/no-drink/" + str(spectrogramMade) + '.png', bbox_inches='tight',
                        pad_inches=0)  # save plot
            # plt.colorbar()
        spectrogramMade = spectrogramMade + 1
#for j in SELECTEDDATE:
#    ddf = tddf[tddf[headers[DATECOL]] == j]
#    for i in headers:
#          #nfft is hamming window, 50
#        namingVar = 0
#        if ddf.columns.get_loc(i) not in SKIP:
#            print i + "is not in skip"
#
#            for imgNum in np.arange(Ltime, Utime,jump):  #for each image from lower bound to upper bound by step size of step
#                plt.cla()  #remove axes
#                plt.clf()  #clear plot to be empty
#                tempddf = ddf  #temporary dataframe for snapshot in time
#                print imgNum  #upper bound for time
#                print imgNum + window  #lower bound for time
#                tempddf = tempddf[tempddf[headers[TIMECOL]] > imgNum]  #remove lower bound of time
#                tempddf = tempddf[tempddf[headers[TIMECOL]] < imgNum + window]  #remove upper bound of time
#                #print "LEN OF tempddf: " + str(len(tempddf['period']))  #size of removed data
#                if len(tempddf) >= 0:
#                    ax = plt.subplot(111) #length,width,height ratio of 1:1:1
#                    Pxx, freqs, bins, im = plt.specgram((tempddf[i]), NFFT=NFFT, noverlap=NOVERLAP, Fs=Fs) #generate spectrogram
#                    plt.savefig(str(i) + str(namingVar) + '.png', bbox_inches='tight', pad_inches = 0) #save plot
#                    plt.colorbar()
#                    namingVar = namingVar + 1
