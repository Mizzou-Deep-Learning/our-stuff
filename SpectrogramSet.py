#from Tkinter import Tk
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
#matplotlib.use("tkAgg")
import matplotlib.pyplot as plt
import math
import random
import os
#matplotlib.use('Agg')
#Tk().withdraw()

# path: the path/destination folder to check if it exists. If it does not exist, create it.
def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise
##########Used for time window of spectrogram###################
HOUR = 1.0
MINUTE = HOUR/60.0
SECOND = MINUTE/60.0
################################################################


##########Vars to allow for computer to see which columns have certain info##### EXCEL COLUMNS ARE 0 BASED##########
TIMECOL = 1 #The column that has the time of day (0-23)
DATECOL = 0 #The column that has the date
DRINKCOL = 7 #The column that has the drink data (0 or 1, 1 being positive for drinking)
SKIP = [DATECOL, TIMECOL, DRINKCOL, 6,8,9,10,11,12,13,14] #used for determining unused columns (such as moods)
################################################################


filename = 'training_mood.csv' #TODO list of csv's instead of single

df = pd.read_csv(filename) #turn excel/csv file into a dataframe (large pandas table)
headers = list(df.columns.values) #list of headers

dfDrinkPts = df[df[headers[DRINKCOL]] == 1] #dataframe of all the instances of drinking

#################Spectrogram vars###############
NFFT = 50 #Window (see Matlab's spectrogram)
Fs = 0.2 #frequency sampling, determines axes
NOVERLAP = 49 #NFFT - 1
PAD_TO = 256 #max(int(pow(2,math.log(1500, 2))), 256)
################################################################

namingVar = 0 #used for naming image files
numDatapoints = 0 #number of datapoints in spectrogram

##################################################################################################
#########################Generate the Drinking Spectrograms#######################################
##################################################################################################

#for each row in the list of drinking data points
#print dfDrinkPts
#print '-----------------------------'
#print df.index[[1,2]]
#df.drop(df.index[[1,2]], inplace=True)

#print '-----------------------------'

#print df
#print df.iloc[2][0]
#print "-----------------"
print dfDrinkPts
#print "@@@@@@@@@@@@@@@@@@@@@@"
#print "A"
for row in xrange(len(dfDrinkPts) -1, -1, -1): # row is the row that has the actual accessible elements/values
    #print row
    #print dfDrinkPts.iloc[row]
    #print dfDrinkPts.iloc[row][0]
    #print dfDrinkPts.iloc[row][1]
    #print dfDrinkPts.iloc[row][2]
    tempdf = df[df[headers[DATECOL]] == dfDrinkPts.iloc[row][DATECOL]]
    #print tempdf
    tempdf = tempdf[tempdf[headers[TIMECOL]] <= (dfDrinkPts.iloc[row][TIMECOL] + 15*MINUTE)]
    tempdf = tempdf[tempdf[headers[TIMECOL]] >= (dfDrinkPts.iloc[row][TIMECOL] - 15*MINUTE)]
    #print str(len(tempdf)) + " : LENGTH"
    #print tempdf
    for j in headers:
        if(tempdf.columns.get_loc(j) not in SKIP):
            #print "LOWEST PART"
            #print tempdf[j]
            #print row
            #print dfDrinkPts.iloc[row]

            PAD_TO = int(pow(2, math.log(len(tempdf[j]), 2))) #calculate the Pad_to for NFFT

            if(numDatapoints < len(tempdf[j])):  # set numDatapoints, used later in finding non-drink spectrograms
                numDatapoints = len(tempdf[j])
            filename = ("Spectrograms/" + str(j)) + "/drink/" + str(namingVar/4)    #cmap=plt.cm.binary,
            Pxx, freqs, bins, im = plt.specgram((tempdf[j]), NFFT=NFFT, noverlap = NOVERLAP, pad_to=PAD_TO,  Fs=Fs)   #generate spectrogram
            make_sure_path_exists("Spectrograms/" + str(j) + "/drink")
            plt.savefig(filename + '.jpg', bbox_inches='tight', pad_inches = 0) #save plot

            tempdf.to_csv(filename + ".txt", sep='\t')
            #np.savetxt(str(j) + "/drink/" + str(namingVar/4) + ".txt", tempdf.to_string)#, fmt = '%d')
            #tfile = open(str(j) + "/drink/" + str(namingVar/4) + '.txt', "w")
            #tfile.write(str(tempdf[j].to_string))
            #tfile.write(len(tempdf[j]))
            #tfile.close()
            namingVar = namingVar + 1
   # df.drop(df.index())
    #print "tempdf:"
    #print tempdf
    #print "range:"
    #print range(i-(numDatapoints-1)/2, i+(numDatapoints-1)/2)
    #print "FOCUSSSSSSSS"
    #print len(df)
    #df = df[df[headers[DATECOL]] != row["period"] or (df[headers[DATECOL] == row["period"]] and (df[df[headers[TIMECOL]] > row["time"] + 15*MINUTE] or df[df[headers[TIMECOL]] < row["time"] - 15*MINUTE])) ]
    index = dfDrinkPts[dfDrinkPts[headers[TIMECOL]] == dfDrinkPts.iloc[row][TIMECOL]].index
    index = index[0]
    #print index
    df = df.drop(df.index[[range(int(index)- (numDatapoints-1)/2,int(index)+(numDatapoints-1)/2)]])
    #print len(df)
    #print "-----"
    #dfDrinkPts = df[df[headers[DRINKCOL]] == 1]
    #print numDatapoints
    #print "LENGTH OF DFDRINKPTS:" + str(len(dfDrinkPts))
    #print str(row)

        #while True:
        #    z = 0
numImages = namingVar/4
spectrogramMade = 0

#print numDatapoints
#print "DONE!!!!!!!!!!!!!!"
dfDrinkPts = df[df[headers[DRINKCOL]] == 1] #dataframe of all the instances of drinking
#print dfDrinkPts
#while True:
#    spectrogramMade = spectrogramMade + 1
##################################################################################################
#########################Generate the No-Drinking Spectrograms####################################
##################################################################################################
df.to_csv("midptDF.txt", sep='\t')

random.seed()
while spectrogramMade != numImages:
    time = random.randrange(0, 23999)
    time = float(time)
    time = time/100.0
    print time
    #print [(df.columns.values[DATECOL])]
    randDate = df.drop_duplicates(subset=[df.columns.values[DATECOL]])
    #print randDate
    #print randDate
    #print len(randDate)
    randomRow = randDate.sample(n=1)
    randomDate = randomRow.iloc[0,0]
    print randomDate
    NoDrinkDf = df[df[headers[DATECOL]] == randomDate]
    #print NoDrinkDf
    #print str(NoDrinkDf[headers[TIMECOL]])
    #print  str(time + 30*MINUTE) + "TIME"
    NoDrinkDf = NoDrinkDf[NoDrinkDf[headers[TIMECOL]] <= (time + 15*MINUTE)]
    NoDrinkDf = NoDrinkDf[NoDrinkDf[headers[TIMECOL]] >= (time - 15*MINUTE)]
    #print NoDrinkDf
    testfordrinkDF = NoDrinkDf[NoDrinkDf[headers[DRINKCOL]] == 1]
    #print testfordrinkDF
    #print len(testfordrinkDF)#.count()
    #print len(NoDrinkDf)
    #print "ABOVE: NODRINKDF LEN"

    if len(testfordrinkDF) == 0 and len(NoDrinkDf) == numDatapoints:
#cmap = plt.cm.binary,
        for j in headers:
            if (tempdf.columns.get_loc(j) not in SKIP):
                ax = plt.subplot(111)  # length,width,height ratio of 1:1:1
                Pxx, freqs, bins, im = plt.specgram((NoDrinkDf[j]),  NFFT=NFFT, pad_to=PAD_TO,  noverlap=NOVERLAP, Fs=Fs)  # generate spectrogram
                print NoDrinkDf[j]
                make_sure_path_exists("Spectrograms/" + str(j) + "/no-drink")
                filename = "Spectrograms/" + str(j) + "/no-drink/" + str(spectrogramMade)
                plt.savefig(filename + '.jpg', bbox_inches='tight',
                        pad_inches=0)  # save plot
            # plt.colorbar()
                NoDrinkDf.to_csv(filename + ".txt", sep='\t')

                #tfile = open(str(j) + "/no-drink/" + str(spectrogramMade) + '.txt', "w")
                #tfile.write(NoDrinkDf[j])
                #tfile.write(len(NoDrinkDf[j]))
                #tfile.close()
        spectrogramMade = spectrogramMade + 1
print spectrogramMade

test = spectrogramMade/4
train = spectrogramMade - test


for j in headers:
    if tempdf.columns.get_loc(j) not in SKIP:
        #make_sure_path_exists(str(j) + "/train.txt")

        #make_sure_path_exists(str(j) + "/test.txt")


        import os

        script_path = os.path.abspath(__file__)  # i.e. /path/to/dir/foobar.py
        script_dir = os.path.split(script_path)[0]  # i.e. /path/to/dir/
        rel_path = "Spectrograms/" + str(j) + "/train.txt"
        abs_file_path = os.path.join(script_dir, rel_path)
        drinktrainfile = open(abs_file_path, "w")

        rel_path = "Spectrograms/" + str(j) + "/test.txt"
        abs_file_path = os.path.join(script_dir, rel_path)

        drinktestfile = open(abs_file_path, "w")

        for i in range(0, test + train):
            if i > test:
                drinktrainfile.write("Spectrograms/" + str(j) + "/drink/" + str(i) + ".jpg drinking\n")
                drinktrainfile.write("Spectrograms/" + str(j) + "/no-drink/" + str(i) + ".jpg no_drinking\n" )

            else:
                drinktestfile.write("Spectrograms/" + str(j) + "/drink/" + str(i) + ".jpg drinking\n")
                drinktestfile.write("Spectrograms/" + str(j) + "/no-drink/" + str(i) + ".jpg no_drinking\n")
        drinktrainfile.close()
        drinktestfile.close()
print "Entering Deep Learning"
execfile('joe.py')
print "Done"

