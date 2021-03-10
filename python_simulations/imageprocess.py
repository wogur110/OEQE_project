import pdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema, argrelmax
import time

## Setting for blurring image
pupil_diameter = 2e-3
eye_length = 24e-3
res_window = 21,21
window_size = 0.2e-3, 0.2e-3
num_color_img_list = 8

#measure time
start_time = time.time()

depthmap = np.load('dataset/depth_image.npy')
image = np.load('dataset/color_image.npy')

focal_length = 1 / (1/(gaze_depth/1000.0) + 1/eye_length)
c = 1   #coefficient for gaussian psf
color_img_list = []
color_img = color_img.astype(float)
depth_img = depth_img.astype(float)
blurred_image = np.zeros_like(color_img)

x,y = np.meshgrid(np.linspace(-window_size[0]/2, window_size[0]/2, res_window[0]), np.linspace(-window_size[1]/2, window_size[1]/2, res_window[1]))    
radius = np.sqrt(x*x + y*y)

depth_list = depth_img[depth_img > 0]

percentiles = np.linspace(0,100,num_color_img_list+1)
depth_bound_list = np.percentile(depth_list, percentiles)

for idx in range(num_color_img_list) :
    pixel_select = np.ones_like(depth_img)
    pixel_select[depth_img < depth_bound_list[idx]] = 0
    pixel_select[depth_img >= depth_bound_list[idx+1]] = 0
    
    depth_select_list = depth_img[pixel_select == 1]
    depth_idx = np.mean(depth_select_list)

    pixel_select = np.stack((pixel_select, pixel_select, pixel_select), axis = 2)
    color_img_idx = color_img * pixel_select

    color_img_list.append((color_img_idx, depth_idx))        


for color_img_idx, depth_idx in color_img_list :
    b = pupil_diameter * abs(eye_length * (1/focal_length - 1 / (depth_idx/1000.0)) - 1) 
    kernel = 2 / (pi * (c * b)**2) * np.exp(-2 * radius**2 / (c * b) ** 2)
    kernel[radius > window_size[0]/2] = 0
    kernel = kernel / np.sum(kernel)

    blurred_image += cv2.filter2D(color_img_idx, -1, kernel)
    #blurred_image += color_img_idx
    print(depth_idx)

pixel_select = np.zeros_like(depth_img)  
pixel_select[depth_img == 0] = 1
pixel_select = np.stack((pixel_select, pixel_select, pixel_select), axis = 2)
color_img_idx = color_img * pixel_select
#blurred_image += color_img_idx

blurred_image = blurred_image / np.max(blurred_image)
print("---{}s seconds---".format(time.time()-start_time))
plt.imshow(blurred_image)

"""
method 1: spatial convolution
"""

"""
method 2: scipy.signal.fftconvolve
"""
# N=512
# depth_image = np.random.random((512,512)) * 8000

# point_y = 300
# point_x = 300

# # get depth planes
# bins = np.arange(0,8000,100)
# depths = np.array([depth_image[point_y][point_x]])
# hist, bins = np.hist(depth_image, bins)
# x = .5*(bins[:,-1]+bins[1:])
# depths = np.append(depths, x[argrelmax(hist,order=4)])
# depths.sort()

# # get 
# image = np.zeros_like(color_image)
# mask = np.zeros_like(color_image)
# for depth in depths:
#     if depths.index(depth) == 0:
#         min_depth = 0
#     else:
#         min_depth = .5 * (depths[depths.index(depth)-1]+ depth)
    
#     if depths.index(depth) == len(depth):
#         max_depth = 8207
#     else:
#         max_depth = .5 * (depths[depths.index(depth)+1]+ depth)

#     for i in range(0:3):
#         mask = color_image * np.double(depth_image >= min_depth and depth_image < max_depth)

        