import pdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema, argrelmax

N=512
depth_image = np.random.random((512,512)) * 8000

point_y = 300
point_x = 300

# get depth planes
bins = np.arange(0,8000,100)
depths = np.array([depth_image[point_y][point_x]])
hist, bins = np.hist(depth_image, bins)
x = .5*(bins[:,-1]+bins[1:])
depths = np.append(depths, x[argrelmax(hist,order=4)])
depths.sort()

# get 
image = np.zeros_like(color_image)
mask = np.zeros_like(color_image)
for depth in depths:
    if depths.index(depth) == 0:
        min_depth = 0
    else:
        min_depth = .5 * (depths[depths.index(depth)-1]+ depth)
    
    if depths.index(depth) == len(depth):
        max_depth = 8207
    else:
        max_depth = .5 * (depths[depths.index(depth)+1]+ depth)

    for i in range(0:3):
        mask = color_image * np.double(depth_image >= min_depth and depth_image < max_depth)

        