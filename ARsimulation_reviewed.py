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

#setting for ScreenInfo
screen_id = 0
screen = screeninfo.get_monitors()[screen_id]
resolution = [screen.width, screen.height]

#setting for PSF
pupil_diameter=2e-3
eye_length=24e-3
eye_relief=1e-1
res_kernel=21

#setting for reduced wiener deconvolution
list_of_slice_numbers=[1,2,4,8,16,32,64,128]

# setting for realsense2 camera
COLOR_CAMERA_MAX_THETA = 41 / 2.0 * (pi / 180)
COLOR_CAMERA_MAX_PHI = 64 / 2.0 * (pi / 180)

# example gaze angle for simulation
converted_theta, converted_phi = COLOR_CAMERA_MAX_THETA/2, 0 

#simulation configure
RES = 640,480

def full_wiener_deconvolution(color_img, depth_img, gaze_depth):
    """
    slice color image by every unique depth in depth image. Create corresponding PSF on each slice. Apply wiener deconvolution on every slice and add up every slice. return normalized reconstructed image.
    """
    focal_length = 1/ (1/gaze_depth + 1/eye_length)
    c=1e+4 #coefficient for gaussian psf
    color_img_list=[]
    color_img=color_img.astype(float)
    depth_img=depth_img.astype(float)
    deconvolved_img=np.zeros_like(color_img)

    x,y=np.meshgrid(np.linspace(-RES[0]//2, RES[0]//2-1,RES[0]), np.linspace(-RES[1]//2,RES[1]//2-1,RES[1]))
    radius=np.sqrt(x*x+y*y)

    depths=np.unique(depth_img[depth_img>0])
    sliced_color_imgs=[]

    for depth in depths:
        pixel_select=np.zeros_like(depth_img)
        pixel_select[depth_img==depth]=1
        pixel_select=np.stack((pixel_select,pixel_select,pixel_select), axis=2)
        color_img_slice = color_img * pixel_select
        sliced_color_imgs.append((color_img_slice, depth/1000.0))

    eye_focal_length = 1/(1/gaze_depth + 1/eye_length)

    for color_img_slice, depth in sliced_color_imgs:
        b = (eye_focal_length/(gaze_depth-eye_focal_length))* pupil_diameter * abs(depth - gaze_depth) / depth # blur diameter
        kernel = np.zeros_like(color_img_slice[:,:,0]) # must be the same size with image for modifying if is_real in skimage.wiener is True
        if b==0:
            kernel[RES[1]//2, RES[0]//2] = 1 # delta function
        else:
            kernel=2/(pi*(c*b)**2)*np.exp(-2*radius**2/(c*b)**2)
            kernel[radius>res_kernel]=0 # use 21*21 nonzero points near origin
        
        #normalization
        if np.sum(kernel)==0: # when does this occurs?
            kernel[res_window[1]//2, res_window[0]//2]=1
        else:
            kernel=kernel/np.sum(kernel)

        R_img_slice, G_img_slice, B_img_slice = color_img_slice[:,:,0], color_img_slice[:,:,1], color_img_slice[:,:,2]
        
        compensate_R = restoration.wiener(R_img_slice, kernel, 1e+0, clip=False) # why is balance 1e+0? why is clip False?
        compensate_G = restoration.wiener(G_img_slice, kernel, 1e+0, clip=False)
        compensate_B = restoration.wiener(B_img_slice, kernel, 1e+0, clip=False)

        compensate_img = np.stack((compensate_R, compensate_G, compensate_B), axis=2)
        deconvolved_img += compensate_img

    # just add original image if depth is zero
    pixel_select = np.zeros_like(depth_img)
    pixel_select[depth_img==0] = 1
    pixel_select=np.stack((pixel_select, pixel_select, pixel_select), axis=2)
    color_img_depth0 = color_img * pixel_select
    deconvolved_img += color_img_depth0
    # clip negative values
    deconvolved_img = np.clip(deconvolved_img, 0, np.max(deconvolved_img))
    deconvolved_img = deconvolved_img/np.sum(deconvolved_img) * np.sum(color_img) / 255.0 # make total brightness similar with original image
    deconvolved_img = np.clip(deconvolved_img,0,1)
    return deconvolved_img

def reduced_wiener_deconvolution(color_img, depth_img, gaze_depth, number_of_slices):
    """
    slice color image by 'number_of_slices' in depth image. Create corresponding PSF on each slice. Apply wiener deconvolution on every slice and add up every slice. return normalized reconstructed image.
    """
    focal_length = 1/ (1/gaze_depth + 1/eye_length)
    c=1e+4 #coefficient for gaussian psf
    color_img_list=[]
    color_img=color_img.astype(float)
    depth_img=depth_img.astype(float)
    deconvolved_img=np.zeros_like(color_img)

    x,y=np.meshgrid(np.linspace(-RES[0]//2, RES[0]//2-1,RES[0]), np.linspace(-RES[1]//2,RES[1]//2-1,RES[1]))
    radius=np.sqrt(x*x+y*y)

    depths = depth_img[depth_img>0]
    percentiles = np.linspace(0,100,number_of_slices+1) # total 
    bounds_of_depth = np.percentile(depths, percentiles, interpolation='nearest')

    depths = np.unique(depths)

    sliced_color_imgs=[]

    for idx in range(number_of_slices): # idx th slice
        pixel_select=np.zeros_like(depth_img)
        for depth in depths: # create boolean mask
            if bounds_of_depth[idx] <= depth and depth< bounds_of_depth[idx+1]:
                pixel_select[depth_img==depth]=1 
        
        if idx == number_of_slices-1: # add last depth on last slice
            pixel_select[depth_img == bounds_of_depth[number_of_slices]] = 1

        masked_depth_img = depth_img[pixel_select == 1]

        if len(masked_depth_img)==0: # if masked_depth_img is blank
            continue
        
        mean_of_depth = np.mean(masked_depth_img)
        
        pixel_select=np.stack((pixel_select,pixel_select,pixel_select), axis=2)
        color_img_slice = color_img * pixel_select
        sliced_color_imgs.append((color_img_slice, mean_of_depth/1000.0))

    eye_focal_length = 1/(1/gaze_depth + 1/eye_length)

    for color_img_slice, depth in sliced_color_imgs:
        b = (eye_focal_length/(gaze_depth-eye_focal_length))* pupil_diameter * abs(depth - gaze_depth) / depth # blur diameter
        kernel = np.zeros_like(color_img_slice[:,:,0]) # must be the same size with image for modifying if is_real in skimage.wiener is True
        if b==0:
            kernel[RES[1]//2, RES[0]//2] = 1 # delta function
        else:
            kernel=2/(pi*(c*b)**2)*np.exp(-2*radius**2/(c*b)**2)
            kernel[radius>res_kernel]=0 # use 21*21 nonzero points near origin
        
        #normalization
        if np.sum(kernel)==0: # when does this occurs?
            kernel[res_window[1]//2, res_window[0]//2]=1
        else:
            kernel=kernel/np.sum(kernel)

        R_img_slice, G_img_slice, B_img_slice = color_img_slice[:,:,0], color_img_slice[:,:,1], color_img_slice[:,:,2]
        
        compensate_R = restoration.wiener(R_img_slice, kernel, 1e+0, clip=False) # why is balance 1e+0? why is clip False?
        compensate_G = restoration.wiener(G_img_slice, kernel, 1e+0, clip=False)
        compensate_B = restoration.wiener(B_img_slice, kernel, 1e+0, clip=False)

        compensate_img = np.stack((compensate_R, compensate_G, compensate_B), axis=2)
        deconvolved_img += compensate_img

    # just add original image if depth is zero
    pixel_select = np.zeros_like(depth_img)
    pixel_select[depth_img==0] = 1
    pixel_select=np.stack((pixel_select, pixel_select, pixel_select), axis=2)
    color_img_depth0 = color_img * pixel_select
    deconvolved_img += color_img_depth0
    # clip negative values
    deconvolved_img = np.clip(deconvolved_img, 0, np.max(deconvolved_img))
    deconvolved_img = deconvolved_img/np.sum(deconvolved_img) * np.sum(color_img) / 255.0 # make total brightness similar with original image
    deconvolved_img = np.clip(deconvolved_img,0,1)

    return deconvolved_img, len(sliced_color_imgs)

if __name__ == "__main__":
    color_image = cv2.imread('imageset/roadimage.png')
    depth_image = cv2.imread('imageset/gray_roaddepthmap.png')
    depth_image = cv2.cvtColor(depth_image, cv2.COLOR_RGB2GRAY)

    color_image = cv2.resize(color_image, dsize = RES, interpolation = cv2.INTER_AREA) # resampling using pixel area relation
    depth_image = cv2.resize(depth_image, dsize = RES, interpolation = cv2.INTER_AREA)

    depth_image = 1.0 / (depth_image + 1) * 20 * 1000 #depth : 10mm ~ 256*10 = 2560mm

    # apply colormap on depth image (image must be converted to 8-bit per pixel first)
    depth_colormap=cv2.applyColorMap(cv2.convertScaleAbs(depth_image,alpha=0.03), cv2.COLORMAP_JET)

    #find tracking point
    H,W = color_image.shape[0], color_image.shape[1]
    point_y = int(H/2*(1 + np.tan(converted_theta)/np.tan(COLOR_CAMERA_MAX_THETA)))
    point_x = int(W/2*(1 + np.tan(converted_phi)/np.tan(COLOR_CAMERA_MAX_PHI)))
    point_y = np.clip(point_y, 0, H-1)
    point_x = np.clip(point_x, 0, W-1)

    reconstructed_image_full = full_wiener_deconvolution(color_image, depth_image, depth_image[point_y][point_x]/1000.0)
    cv2.imwrite("Dataset/wiener_img.png", (reconstructed_image_full*255).astype(int))

    f = open("Dataset/simulation_result.txt","w")

    for num_of_slice in list_of_slice_numbers:
        mean_time = 0.0
        actual_num_of_slice = 0
        reconstructed_image_reduced = np.zeros_like(color_image)
        
        for i in range(10): # repeat same work for 10 times and calculate mean time
            time_0 = time.time()
            reconstructed_image_reduced, actual_num_of_slice = reduced_wiener_deconvolution(color_image, depth_image, depth_image[point_y][point_x] / 1000.0, num_of_slice)
            time_1 = time.time()

            mean_time += time_1 - time_0
        
        mean_time /= 10

        print("actual_num_of_slice :", actual_num_of_slice)
        print("mean computation time : ", mean_time)
        PSNR = metrics.peak_signal_noise_ratio(reconstructed_image_full, reconstructed_image_reduced)
        SSIM = metrics.structural_similarity(reconstructed_image_full, reconstructed_image_reduced, multichannel=True)
        print("PSNR : ", PSNR)
        print("SSIM : ", SSIM)

        writeline = str(actual_num_of_slice) + " " + str(mean_time) + " " + str(PSNR) + " " + str(SSIM) + "\n"
        f.write(writeline)
        
        reconstructed_image_reduced = cv2.circle(reconstructed_image_reduced, (point_x, point_y), 3, (0,255,0), -1)
        cv2.namedWindow('reconstructed_image_reduced', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('reconstructed_image_reduced', reconstructed_image_reduced)
        cv2.waitKey(1)

        cv2.imwrite("Dataset/reconstructed_image_"+str(actual_num_of_slice)+".png", (reconstructed_image_reduced * 255).astype(int))
        

    f.close()
