import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import gridspec
from matplotlib.font_manager import FontProperties
from pylab import *
from sklearn.neighbors import KernelDensity

import joypy
import pandas as pd
from matplotlib import cm

# fonts
rc('text', usetex=True)
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})

rc('text', usetex=True)
#rcParams['text.latex.preamble']=[r"\usepackage[helvet]{sfmath}"]

rcParams['text.latex.preamble'] = r"\usepackage[helvet]{sfmath}"

x1 = np.loadtxt("histogram_entangled_0layers.txt")

x2 = np.loadtxt("histogram_entangled_2layers.txt")

x3 = np.loadtxt("histogram_entangled_4layers.txt")

x4 = np.loadtxt("histogram_entangled_6layers.txt")




c = np.loadtxt("Cop_4x4.txt")
c = np.sort(c)
c = c[0:600]

gs = 0.0

#x1 = x1[0:200]
#x2 = x2[0:200]
#x3 = x3[0:200]
#x4 = x4[0:200]
#x5 = x5[0:200]
x6 = c

plt.rc("font", size=18)
x = [x1-gs,x2-gs,x3-gs,x4-gs,x6-gs]
#x = [x1-gs,x2-gs,x3-gs,x4-gs]
colors = [ 'deepskyblue', 'mediumseagreen',"orangered","midnightblue","violet"]
#colors = [ 'deepskyblue', 'mediumseagreen',"orangered","midnightblue"]
fig, ax = joypy.joyplot(x,labels= ["p. state", "1 lay","2 lay","3 lay", "C"],overlap=0.4,colormap=cm.viridis_r,grid=True,alpha=0.4,color=colors)

#plt.ylabel("Frequency")

ax[-1].set_xlabel(r"$E(\theta)$")

#for a in ax[:-1]:
#    a.set_xlim([-0.01,0.152])
plt.rc("font", size=18)
plt.subplots_adjust(bottom=0.15)


plt.savefig('KDEhist_NO_FC.pdf')

