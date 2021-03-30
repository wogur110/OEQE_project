## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
##      Open CV and Numpy integration        ##
###############################################
import pdb
import sys
import pyrealsense2 as rs
import numpy as np
import cv2
from math import pi
import time

import zmq
from msgpack import loads
import threading

import numpy as np
import math
import time
import scipy.io
from sklearn.linear_model import LinearRegression
from scipy import fftpack

from skimage import color, data, restoration, metrics

import screeninfo

#Get ScreenInfo
screen_id = 0
screen = screeninfo.get_monitors()[screen_id]
resolution = [screen.width, screen.height]

## Setting for blurring image
pupil_diameter = 2e-3
eye_length = 24e-3
eye_relief = 1e-1
kernel_radius_pixel = 21
#num_color_img_list = 8    # max num in simulation : 256
num_color_img_list_list = [1,2,4,8,16,32,64,128,256,512,1024,2048]

#simulation configure
RES = 640,480

COLOR_CAMERA_MAX_THETA = 41 / 2.0 * (pi / 180)
COLOR_CAMERA_MAX_PHI = 64 / 2.0 * (pi / 180)
converted_theta, converted_phi = 0, 0

def full_blurring_image(color_img, depth_img, gaze_depth) :
    focal_length = 1 / (1/(gaze_depth) + 1/eye_length)
    c = 0.5e+5 #coefficient for gaussian psf
    color_img_list = []
    color_img = color_img.astype(float)
    depth_img = depth_img.astype(float)
    blurred_image = np.zeros_like(color_img)

    x,y = np.meshgrid(np.linspace(-RES[0]//2, RES[0]//2 - 1, RES[0]), np.linspace(-RES[1]//2, RES[1]//2 - 1, RES[1]))
    radius = np.sqrt(x*x + y*y)

    #For Full deconvolution
    depth_list = np.unique(depth_img[depth_img > 0])
    for depth in depth_list :
        pixel_select = np.zeros_like(depth_img)
        pixel_select[depth_img == depth] = 1
        depth_select_list = depth_img[pixel_select == 1]

        pixel_select = np.stack((pixel_select, pixel_select, pixel_select), axis = 2)
        color_img_idx = color_img * pixel_select

        color_img_list.append((color_img_idx, depth / 1000.0))


    eye_focal_length = 1 / (1 / gaze_depth + 1 / eye_length)

    for color_img_idx, depth_idx in color_img_list :
        b = (eye_focal_length / (gaze_depth - eye_focal_length)) * pupil_diameter * abs(depth_idx - gaze_depth) / depth_idx

        kernel = np.zeros_like(color_img_idx[:,:,0])
        if b == 0 :
            kernel[RES[1]//2, RES[0]//2] = 1
        else :
            kernel = 2 / (pi * (c * b)**2) * np.exp(-2 * radius**2 / (c * b)**2)
            kernel[radius > kernel_radius_pixel] = 0    #Use 21*21 nonzero points near origin, otherwise, value is zero

        if np.sum(kernel) == 0 :    
            kernel[res_window[1]//2, res_window[0]//2] = 1            
        else :
            kernel = kernel / np.sum(kernel)

        R_img_idx, G_img_idx, B_img_idx = color_img_idx[:,:,0], color_img_idx[:,:,1], color_img_idx[:,:,2]

        compensate_R = restoration.wiener(R_img_idx, kernel, 1e+1, clip=False)
        compensate_G = restoration.wiener(G_img_idx, kernel, 1e+1, clip=False)
        compensate_B = restoration.wiener(B_img_idx, kernel, 1e+1, clip=False)

        compensate_img = np.stack((compensate_R, compensate_G, compensate_B), axis = 2)
        blurred_image += compensate_img

        #print("depth idx : ", depth_idx)
        # print("kernel max :", kernel.max())

    pixel_select = np.zeros_like(depth_img)
    pixel_select[depth_img == 0] = 1
    pixel_select = np.stack((pixel_select, pixel_select, pixel_select), axis = 2)
    color_img_zero_depth = color_img * pixel_select
    blurred_image += color_img_zero_depth  #just add zero depth pixel to blurred_image

    blurred_image = blurred_image / np.max(blurred_image)
    blurred_image = np.clip(blurred_image,0,1)
    return blurred_image



def blurring_image(color_img, depth_img, gaze_depth, num_color_img_list) :
    focal_length = 1 / (1/(gaze_depth) + 1/eye_length)
    c = 0.5e+5 #coefficient for gaussian psf
    color_img_list = []
    color_img = color_img.astype(float)
    depth_img = depth_img.astype(float)
    blurred_image = np.zeros_like(color_img)

    x,y = np.meshgrid(np.linspace(-RES[0]//2, RES[0]//2 - 1, RES[0]), np.linspace(-RES[1]//2, RES[1]//2 - 1, RES[1]))
    radius = np.sqrt(x*x + y*y)

    depth_list = depth_img[depth_img > 0]

    percentiles = np.linspace(0,100,num_color_img_list+1)
    depth_bound_list = np.percentile(depth_list, percentiles)

    for idx in range(num_color_img_list) :
        pixel_select = np.ones_like(depth_img)
        pixel_select[depth_img < depth_bound_list[idx]] = 0
        pixel_select[depth_img >= depth_bound_list[idx+1]] = 0
        if idx == num_color_img_list - 1 :
            pixel_select[depth_img == depth_bound_list[idx+1]] = 1

        depth_select_list = depth_img[pixel_select == 1]
        if len(depth_select_list) == 0 :
            continue
        depth_idx = np.mean(depth_select_list)
        if int(gaze_depth * 1000) in depth_select_list :
            depth_idx = int(gaze_depth * 1000)

        pixel_select = np.stack((pixel_select, pixel_select, pixel_select), axis = 2)
        color_img_idx = color_img * pixel_select

        color_img_list.append((color_img_idx, depth_idx / 1000.0))


    eye_focal_length = 1 / (1 / gaze_depth + 1 / eye_length)

    for color_img_idx, depth_idx in color_img_list :
        b = (eye_focal_length / (gaze_depth - eye_focal_length)) * pupil_diameter * abs(depth_idx - gaze_depth) / depth_idx

        kernel = np.zeros_like(color_img_idx[:,:,0])
        if b == 0 :
            kernel[RES[1]//2, RES[0]//2] = 1
        else :
            kernel = 2 / (pi * (c * b)**2) * np.exp(-2 * radius**2 / (c * b)**2)
            kernel[radius > kernel_radius_pixel] = 0    #Use 21*21 nonzero points near origin, otherwise, value is zero

        if np.sum(kernel) == 0 :    
            kernel[res_window[1]//2, res_window[0]//2] = 1            
        else :
            kernel = kernel / np.sum(kernel)

        R_img_idx, G_img_idx, B_img_idx = color_img_idx[:,:,0], color_img_idx[:,:,1], color_img_idx[:,:,2]

        compensate_R = restoration.wiener(R_img_idx, kernel, 1e+1, clip=False)
        compensate_G = restoration.wiener(G_img_idx, kernel, 1e+1, clip=False)
        compensate_B = restoration.wiener(B_img_idx, kernel, 1e+1, clip=False)

        compensate_img = np.stack((compensate_R, compensate_G, compensate_B), axis = 2)
        blurred_image += compensate_img

        #print("depth idx : ", depth_idx)
        # print("kernel max :", kernel.max())

    pixel_select = np.zeros_like(depth_img)
    pixel_select[depth_img == 0] = 1
    pixel_select = np.stack((pixel_select, pixel_select, pixel_select), axis = 2)
    color_img_zero_depth = color_img * pixel_select
    blurred_image += color_img_zero_depth  #just add zero depth pixel to blurred_image

    blurred_image = blurred_image / np.max(blurred_image)
    blurred_image = np.clip(blurred_image,0,1)
    return blurred_image, len(color_img_list)


if __name__ == "__main__":
    #load simulation image for simulation
    color_image = cv2.cvtColor(cv2.imread('imageset/roadimage.png'), cv2.COLOR_BGR2RGB) 
    depth_image = cv2.cvtColor(cv2.imread('imageset/gray_roaddepthmap.png'), cv2.COLOR_BGR2RGB)
    depth_image = cv2.cvtColor(depth_image, cv2.COLOR_RGB2GRAY)

    color_image = cv2.resize(color_image, dsize = RES, interpolation = cv2.INTER_AREA)
    depth_image = cv2.resize(depth_image, dsize = RES, interpolation = cv2.INTER_AREA)

    depth_image = (256 - depth_image) * 10 #depth : 10mm ~ 256*10 = 2560mm

    # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

    # Find tracking point
    H, W = color_image.shape[0], color_image.shape[1]
    point_y = int(H/2 + H/2 * (np.tan(converted_theta) / np.tan(COLOR_CAMERA_MAX_THETA)))
    point_x = int(W/2 + W/2 * (np.tan(converted_phi) / np.tan(COLOR_CAMERA_MAX_PHI)))
    point_y = np.clip(point_y, 0, H-1)
    point_x = np.clip(point_x, 0, W-1)

    full_blurred_image = full_blurring_image(color_image, depth_image, depth_image[point_y][point_x] / 1000.0)
    cv2.imwrite("Dataset/full_blurred_img.png", (full_blurred_image * 255).astype(int))

    f = open("Dataset/simulation_result.txt", "w")

    for num_color_img_list in num_color_img_list_list :
        mean_time = 0.0
        real_num_color_img_list = 0
        blurred_image = np.zeros_like(color_image)
        for i in range(10) :
            time_0 = time.time()
            blurred_image, real_num_color_img_list = blurring_image(color_image, depth_image, depth_image[point_y][point_x] / 1000.0, num_color_img_list)
            time_1 = time.time()
        
            mean_time += time_1 - time_0
        mean_time /= 10

        print("num_color_img_list :", real_num_color_img_list)
        print("mean computation time : ", mean_time)
        PSNR = metrics.peak_signal_noise_ratio(full_blurred_image, blurred_image)
        SSIM = metrics.structural_similarity(full_blurred_image, blurred_image, multichannel=True)
        print("PSNR : ", PSNR)
        print("SSIM : ", SSIM)

        writeline = str(real_num_color_img_list) + " " + str(mean_time) + " " + str(PSNR) + " " + str(SSIM) + "\n"
        f.write(writeline)
        
        blurred_image = cv2.circle(blurred_image, (point_x, point_y), 3, (0,255,0), -1)
        cv2.namedWindow('Convert_blurred_image', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Convert_blurred_image', blurred_image)
        cv2.waitKey(1)

        cv2.imwrite("Dataset/blurred_img_num_"+str(real_num_color_img_list)+".png", (blurred_image * 255).astype(int))
        

    f.close()