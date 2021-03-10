import pdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema, argrelmax, fftconvolve, convolve
import time
import cv2
from math import pi


## Setting for blurring image
pupil_diameter = 2e-3
eye_length = 24e-3
res_window = 21,21
window_size = 0.2e-3, 0.2e-3
num_color_img_list = 8

depth_img = np.load('dataset/depth_image.npy')
color_img = np.load('dataset/color_image.npy')

point_y = 300
point_x = 300

gaze_depth = depth_img[point_y][point_x]

"""
method 1: spatial convolution
"""
#measure time
start_time = time.time()

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
    #print(depth_idx)

pixel_select = np.zeros_like(depth_img)  
pixel_select[depth_img == 0] = 1
pixel_select = np.stack((pixel_select, pixel_select, pixel_select), axis = 2)
color_img_idx = color_img * pixel_select
#blurred_image += color_img_idx

blurred_image = blurred_image / np.max(blurred_image)
print("---{}s seconds---".format(time.time()-start_time))
plt.imshow(blurred_image)



"""
method 2: scipy.signal.fftconvolve
"""
#measure time
start_time = time.time()

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
    kernel = np.stack([kernel,kernel,kernel],axis=2)
    blurred_image += fftconvolve(color_img_idx, kernel,mode='same')
    #blurred_image += color_img_idx
    #print(depth_idx)

pixel_select = np.zeros_like(depth_img)  
pixel_select[depth_img == 0] = 1
pixel_select = np.stack((pixel_select, pixel_select, pixel_select), axis = 2)
color_img_idx = color_img * pixel_select
#blurred_image += color_img_idx

blurred_image = blurred_image / np.max(blurred_image)
print("---{}s seconds---".format(time.time()-start_time))
plt.imshow(blurred_image)
