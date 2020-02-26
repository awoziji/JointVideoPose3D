import matplotlib.pyplot as plt
from matplotlib import collections  as mc
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np

data = np.array([[[0, 0, 1], [1, 0, 0]]])

lc = Line3DCollection(data)
fig = plt.figure()
ax = fig.gca(projection='3d')

points = ax.scatter(data[:,0],
                    data[:,1],
                    data[:,2],
                    color='green',
                    edgecolors='yellow', zorder=10)

ax.add_collection(lc)
ax.autoscale()
ax.margins(0.1)
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')

plt.grid(True)
