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

fields =['period','time','HR']
df = pd.read_csv(filename, usecols=fields)

print df


