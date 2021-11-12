from matplotlib import pyplot as plt
from matplotlib.text import TextPath
import pandas as pd
from glob import glob
from numpy.random import default_rng
plt.style.use(['tableau-colorblind10'])
SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE) 
#plt.rcParams.update({
#    "text.usetex": True,
#    "font.family": "sans-serif",
#    "font.sans-serif": ["Helvetica"]})

class FancyPlot:
    def __init__(self,folder):
        self.folder_ = folder
        self.path_ = self.path(folder)
        self.df_ = self.readCsv(self.path_)

    def readCsv(self, path):
        return pd.read_csv(path)

    def path(self, folder):
        return glob(folder + "/*.csv")[0] 

    def generateMarkers(self,markerName):
        return TextPath((0,0), markerName)
    
    """
     TODO make a box around each word
     TODO markersize does that all words take up the same space. I want that all words/letters have the same font size
    """
    def plot(self):
        rng = default_rng(seed=7) #make sure we get the same numbers every time
        fig1 = plt.figure(facecolor='white')
        ax1 = plt.axes()
        for index, row in self.df_.iterrows():
            ax1.plot(row["Differanse"],rng.random(), marker = self.generateMarkers(row["Hobby"]), markersize = len(row["Adjektiv"])*6, mew=0.4) 
        
        ax1.axes.get_yaxis().set_visible(False)
        ax1.set_xlim([-1, 1])
        ax1.set_ylim([-0.1, 1.1])
        ax1.set_xlabel("Difference")
        plt.savefig("figs/_favorite_hobby_is_.eps")
        plt.show()

if __name__== "__main__":
    s = FancyPlot("data")
    s.plot()
