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
from scipy.linalg import block_diag

#setting for ScreenInfo
screen_id = 0
screen = screeninfo.get_monitors()[screen_id]
resolution = [screen.width, screen.height]

#setting for PSF
pupil_diameter = 3e-3
eye_length = 20e-3
eye_relief = 1e-1
kernel_radius_pixel = 21

#setting for reduced wiener deconvolution
num_slicing_imgs_list=[1,2,4,8,16,32,64,128,256,512,1024,2048]

# setting for realsense2 camera
COLOR_CAMERA_MAX_THETA = 30 / 2.0 * (pi / 180)
COLOR_CAMERA_MAX_PHI = 30 / 2.0 * (pi / 180)

# example gaze angle for simulation
converted_theta, converted_phi = 0, 0 

#simulation configure
RES = 500,500
acc_depth = 3.0 #accommodation depth : 3000mm

def inverse_filtering(color_img, depth_img, gaze_depth, num_slicing_imgs):
    """
    slice color image by 'num_slicing_imgs' in depth image. Create corresponding PSF on each slice. Apply wiener deconvolution on every slice and add up every slice. return normalized reconstructed image.
    """
    eye_focal_length = 1 / (1 / gaze_depth + 1 / eye_length)
    c = 1e+4 # coefficient for gaussian psf
    color_img=color_img.astype(float)
    depth_img=depth_img.astype(float)
    filtered_img=np.zeros_like(color_img)
    edge = np.zeros_like(depth_img)

    # Calculate target intensity sum
    target_intensity_sum = np.sum(color_img)

    x,y = np.meshgrid(np.linspace(-RES[0]//2, RES[0]//2-1,RES[0]), np.linspace(-RES[1]//2,RES[1]//2-1,RES[1]))
    radius = np.sqrt(x*x+y*y)

    depths = depth_img[depth_img>0]
    percentiles = np.linspace(0,100,num_slicing_imgs+1)
    depth_bounds = np.percentile(depths, percentiles, interpolation='nearest')
    depths = np.unique(depths)

    sliced_color_imgs = []

    for idx in range(num_slicing_imgs): # idx th slice
        pixel_select=np.zeros_like(depth_img)
        for depth in depths: # create boolean mask
            if depth_bounds[idx] <= depth and depth< depth_bounds[idx+1]:
                pixel_select[depth_img==depth] = 1 
        
        if idx == num_slicing_imgs - 1 : # add last depth on last slice
            pixel_select[depth_img == depth_bounds[num_slicing_imgs]] = 1

        masked_depth_img = depth_img[pixel_select == 1]

        if len(masked_depth_img) == 0: # if masked_depth_img is blank
            continue
        
        mean_depth = np.mean(masked_depth_img)
        edge += cv2.Canny(np.uint8(pixel_select*255), 50, 100)
        pixel_select = np.stack((pixel_select,pixel_select,pixel_select), axis = 2)
        sliced_color_img = color_img * pixel_select
        sliced_color_imgs.append((sliced_color_img, mean_depth / 1000.0))

    for sliced_color_img, mean_depth in sliced_color_imgs:
        b = (eye_focal_length / (gaze_depth - eye_focal_length))* pupil_diameter * abs(mean_depth - gaze_depth) / mean_depth # blur diameter
        kernel = np.zeros_like(sliced_color_img[:,:,0]) # must be the same size with image for modifying if is_real in skimage.wiener is True
        
        if b == 0 :
            kernel[RES[1]//2, RES[0]//2] = 1 # delta function
        else :
            kernel = 2 / (pi * (c * b)**2) * np.exp(-2 * radius**2 / (c * b)**2)
            kernel[radius > kernel_radius_pixel] = 0    #Use 21*21 nonzero points near origin, otherwise, value is zero
        
        #normalization
        if np.sum(kernel) == 0: # when does this occurs?
            kernel[res_window[1]//2, res_window[0]//2]=1
        else:
            kernel = kernel / np.sum(kernel)

        R_img_slice, G_img_slice, B_img_slice = sliced_color_img[:,:,0], sliced_color_img[:,:,1], sliced_color_img[:,:,2]
        
        compensate_R = restoration.wiener(R_img_slice, kernel, 0.1e+0, clip=False) # why is balance 1e+0? why is clip False?
        compensate_G = restoration.wiener(G_img_slice, kernel, 0.1e+0, clip=False)
        compensate_B = restoration.wiener(B_img_slice, kernel, 0.1e+0, clip=False)

        compensate_img = np.stack((compensate_R, compensate_G, compensate_B), axis=2)
        filtered_img += compensate_img

    #just add zero depth pixel to filtered image
    pixel_select = np.zeros_like(depth_img)
    pixel_select[depth_img==0] = 1
    pixel_select = np.stack((pixel_select, pixel_select, pixel_select), axis = 2)
    color_img_zero_depth = color_img * pixel_select
    filtered_img += color_img_zero_depth

    edge = np.clip(edge, 0, 255).astype('uint8')
    dilated_edge = cv2.dilate(edge, np.ones((3, 3)))
    dilated_edge = np.stack((dilated_edge, dilated_edge, dilated_edge), axis=2)

    #blurred_filtered_img = cv2.GaussianBlur(filtered_img, (5, 5), 0) # Smoothing boundary
    blurred_filtered_img = filtered_img # No smoothing boundary
    smoothed_filtered_img = np.where(dilated_edge==np.array([255,255,255]), blurred_filtered_img, filtered_img)

    #smoothed_filtered_img = filtered_img
    smoothed_filtered_img = np.clip(smoothed_filtered_img, 0, np.max(smoothed_filtered_img))
    smoothed_filtered_img = smoothed_filtered_img / np.sum(smoothed_filtered_img) * target_intensity_sum / 255.0
    smoothed_filtered_img = np.clip(smoothed_filtered_img, 0, 1)

    return smoothed_filtered_img, len(sliced_color_imgs)

def display2retina(smoothed_filtered_img, acc_depth) :
    # Retina
    rdx = 0.02e-3 # distance of adjacent retina coordinates

    r = (rdx * RES[0] / 2.0 / eye_length) / np.tan(COLOR_CAMERA_MAX_PHI) # crop ratio from display img to retina img 
    retina_img = smoothed_filtered_img[int(RES[1] * (1-r) / 2): int(RES[1] * (1+r) / 2) , int(RES[0] * (1-r) / 2) : int(RES[0] * (1+r) / 2)]
    retina_img = cv2.resize(retina_img, dsize = RES)

    retina_img = (retina_img * 255).astype(np.uint8)
    retina_img = np.clip(retina_img, 0, 255)

    return retina_img

def display2retina_LF(smoothed_filtered_img, acc_depth) :
    '''
    acc_depth : accommodation depth
    '''
    # Display
    size_display = [2 * acc_depth * np.tan(COLOR_CAMERA_MAX_PHI), 2 * acc_depth * np.tan(COLOR_CAMERA_MAX_THETA)]    
    dx,dy = np.meshgrid(np.linspace(-size_display[0]/2.0, size_display[0]/2.0, RES[0]), np.linspace(-size_display[1]/2.0, size_display[1]/2.0, RES[1]))

    # Pupil
    Res_pupil = [7,7]
    px, py = np.meshgrid(np.linspace(-pupil_diameter/2.0, pupil_diameter/2.0, Res_pupil[0]), np.linspace(-pupil_diameter/2.0, pupil_diameter/2.0, Res_pupil[1]))
    pupil = (np.sqrt(px*px+py*py) <= pupil_diameter / 2.0)

    # Retina
    rdx = rdy = 0.02e-3
    rx, ry = np.meshgrid(np.linspace(-RES[0]/2.0, RES[0]/2.0, RES[0]) * rdx, np.linspace(-RES[1]/2.0, RES[1]/2.0, RES[1]) * rdy)

    # Ray initialization
    res_ray = Res_pupil[0] * Res_pupil[1] * RES[0] * RES[1]
    rays = np.zeros((res_ray,9))
    for py_idx in range(Res_pupil[1]) :
        for px_idx in range(Res_pupil[0]) :
            p_idx = py_idx * Res_pupil[0] + px_idx
            
            for ry_idx in range(RES[1]) :
                for rx_idx in range(RES[0]) :
                    retina_idx = ry_idx * RES[0] + rx_idx
                    idx = p_idx * RES[0] * RES[1] + retina_idx

                    rays[idx][0] = rx[ry_idx][rx_idx]
                    rays[idx][1] = np.arctan((px[py_idx][px_idx] - rx[ry_idx][rx_idx]) / eye_length)
                    rays[idx][2] = ry[ry_idx][rx_idx]
                    rays[idx][3] = np.arctan((py[py_idx][px_idx] - ry[ry_idx][rx_idx]) / eye_length)
                    rays[idx][4] = rx_idx
                    rays[idx][5] = ry_idx
                    rays[idx][6] = px_idx
                    rays[idx][7] = py_idx
                    rays[idx][8] = pupil[py_idx][px_idx]

    focal_length = 1 / (1 / eye_length + 1 / acc_depth)
    # Ray transfer matrix
    transfer_x = np.dot(np.array([[1, acc_depth],[0, 1]]), np.dot(np.array([[1, 0], [-1 / focal_length, 1]]), np.array([[1, eye_length], [0, 1]])))
    transfer_y = transfer_x

    transfer = block_diag(transfer_x, transfer_y, np.eye(5))

    ray_on_retina = np.dot(transfer, rays.T).T
    retina_img = np.zeros((RES[1], RES[0], 3))

    for idx in range(res_ray) :
        ray = ray_on_retina[idx]
        dx = ray[0]; dy = ray[2];
        rx_idx = int(ray[4]); ry_idx = int(ray[5]); px_idx = int(ray[6]); py_idx = int(ray[7]);

        wx_idx = int(dx / size_display[0] * (RES[0] - 1) + RES[0] / 2 - 0.5)
        wy_idx = int(dy / size_display[1] * (RES[1] - 1) + RES[1] / 2 - 0.5)

        if wx_idx in range(RES[0]) and wy_idx in range(RES[1]) and pupil[py_idx][px_idx]:
            retina_img[ry_idx][rx_idx] += smoothed_filtered_img[wy_idx][wx_idx]

    retina_img = retina_img / np.count_nonzero(pupil)
    retina_img = cv2.rotate(retina_img, cv2.ROTATE_180)

    retina_img = (retina_img * 255).astype(np.uint8)
    retina_img = np.clip(retina_img, 0, 255)

    return retina_img

        

if __name__ == "__main__":
    color_image = cv2.imread('imageset/Castle/Lightfield/0025.png')
    depth_image = cv2.imread('imageset/Castle/Depthmap/depthmap.png')    
    depth_image = cv2.cvtColor(depth_image, cv2.COLOR_BGR2GRAY)
    LF_rendered_image = cv2.imread('result/LF_result/Castle/depth%d.png' %(int(acc_depth *1000))) #rendered which accomodation depth is 3000mm

    LF_rendered_image = cv2.resize(LF_rendered_image, dsize = RES, interpolation = cv2.INTER_AREA) # resampling using pixel area relation
    color_image = cv2.resize(color_image, dsize = RES, interpolation = cv2.INTER_AREA) 
    depth_image = cv2.resize(depth_image, dsize = RES, interpolation = cv2.INTER_AREA)

    depth_image = (255 - depth_image) * 3.0 / 255.0 * 1000.0 #depth : 0m(255) ~ 3m(0) linearly distributed

    # apply colormap on depth image (image must be converted to 8-bit per pixel first)
    depth_colormap=cv2.applyColorMap(cv2.convertScaleAbs(depth_image,alpha=0.03), cv2.COLORMAP_JET)
    
    # H, W = color_image.shape[0], color_image.shape[1]
    # point_y = int(H/2 * (1 + np.tan(converted_theta) / np.tan(COLOR_CAMERA_MAX_THETA)))
    # point_x = int(W/2 * (1 + np.tan(converted_phi) / np.tan(COLOR_CAMERA_MAX_PHI)))
    # point_y = np.clip(point_y, 0, H-1)
    # point_x = np.clip(point_x, 0, W-1)

    #find tracking point where depth is 3000mm
    point_y = np.where(depth_image == acc_depth * 1000)[0][0]
    point_x = np.where(depth_image == acc_depth * 1000)[1][0]

    f = open("result/AR_image_rendering/simulation_result.txt","w") # write simulation result

    for num_slicing_imgs in num_slicing_imgs_list:
        mean_time = 0.0
        actual_num_slicing_imgs = 0
        smoothed_filtered_img = np.zeros_like(color_image)
        
        for i in range(5): # repeat same work for 5 times and calculate mean time
            time_0 = time.time()
            smoothed_filtered_img, actual_num_slicing_imgs = inverse_filtering(color_image, depth_image, depth_image[point_y][point_x] / 1000.0, num_slicing_imgs)

            mean_time += time.time() - time_0
        
        mean_time /= 5

        rendered_retina_img = display2retina(smoothed_filtered_img, acc_depth)

        print("actual_num_slicing_imgs :", actual_num_slicing_imgs)
        print("mean computation time : ", mean_time)
        PSNR = metrics.peak_signal_noise_ratio(LF_rendered_image, rendered_retina_img)
        SSIM = metrics.structural_similarity(LF_rendered_image, rendered_retina_img, multichannel=True)
        print("PSNR : ", PSNR)
        print("SSIM : ", SSIM)

        writeline = str(actual_num_slicing_imgs) + " " + str(mean_time) + " " + str(PSNR) + " " + str(SSIM) + "\n"
        f.write(writeline)
        
        cv2.namedWindow('smoothed_filtered_img', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('smoothed_filtered_img', smoothed_filtered_img)
        cv2.waitKey(1)

        rendered_retina_img = cv2.circle(rendered_retina_img, (point_x, point_y), 2, (0,0,255), -1)
        cv2.namedWindow('rendered_retina_img', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('rendered_retina_img', rendered_retina_img)
        cv2.waitKey(1)

        cv2.namedWindow('LF_rendered_image', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('LF_rendered_image', LF_rendered_image)
        cv2.waitKey(1)

        cv2.imwrite("result/AR_image_rendering/rendered_retina_img_"+str(actual_num_slicing_imgs)+".png", rendered_retina_img)
       
    f.close()
