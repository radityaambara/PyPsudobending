from all_func import VelocityGrid
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import LinearNDInterpolator

"""
This is example of how to use psudobending function in velgrid class.
"""
modvel = pd.read_csv("vel_example.csv")
node = np.vstack((modvel.X, modvel.Y, modvel.Z)).T
vel_node_P = np.array(modvel.Vp)
delt = 0.25
deltn = 0.5
xfac = 1.2
iter1 = 50
iter2 = 50
tmin = 0.0001
iteration_number = 10
xmin = np.amin(modvel.X)
xmax = np.amax(modvel.X)
ymin = np.amin(modvel.Y)
ymax = np.amax(modvel.Y)
zmin = np.amin(modvel.Z)
zmax = np.amax(modvel.Z)

#interpolate velocity model into regular grid
xnode = np.arange(xmin, xmax + deltn, deltn)
ynode = np.arange(ymin, ymax + deltn, deltn)
znode = np.arange(zmin, zmax + deltn, deltn)
Xinter, Yinter, Zinter = np.meshgrid(xnode, ynode, znode, indexing='ij')
interpP = LinearNDInterpolator(node, vel_node_P)
gridVp = np.round_(interpP(Xinter, Yinter, Zinter), decimals=4)
# determine velgrid object
velgridP = VelocityGrid(node, xnode, ynode, znode, gridVp, deltn, delt, xfac, iter1, iter2, tmin)

#create source and receiver position.
source=np.array([329.15998475360055,9084.352804076592,-6.791])
receiver=np.array([327.62272549612493,9082.795913064188,1.535])
#this will be the initial and final path position
path = np.vstack((source, receiver))

#bend the raypath using psudobending and calculate the traveltime using psudobending function
psudopath,tt=velgridP.psudobending(path)

print('traveltime is',tt)
fig = plt.figure()
ax = Axes3D(fig)
ax.plot(psudopath[:,0],psudopath[:,1],psudopath[:,2],alpha=1)
ax.scatter(source[0],source[1],source[2])
ax.scatter(receiver[0],receiver[1],receiver[2])
#ax.scatter(node[:,0],node[:,1],node[:,2])
plt.show()

