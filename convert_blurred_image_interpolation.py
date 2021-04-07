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

from skimage import color, data, restoration, metrics

import screeninfo

#Get ScreenInfo
screen_id = 1
screen = screeninfo.get_monitors()[screen_id]
resolution = [screen.width, screen.height]

## Setting for realsense
REALSENSE_CAMERA = "D435" #D435 / L515

# Camera setting and tracking setting
if REALSENSE_CAMERA == "D435" :
    DEPTH_CAMERA_MAX_THETA = 57 / 2.0 * (pi / 180)
    DEPTH_CAMERA_MAX_PHI = 86 / 2.0 * (pi / 180)
    COLOR_CAMERA_MAX_THETA = 41 / 2.0 * (pi / 180)
    COLOR_CAMERA_MAX_PHI = 64 / 2.0 * (pi / 180)
elif REALSENSE_CAMERA == "L515" :
    DEPTH_CAMERA_MAX_THETA = 55 / 2.0 * (pi / 180)
    DEPTH_CAMERA_MAX_PHI = 70 / 2.0 * (pi / 180)
    COLOR_CAMERA_MAX_THETA = 43 / 2.0 * (pi / 180)
    COLOR_CAMERA_MAX_PHI = 70 / 2.0 * (pi / 180)

if REALSENSE_CAMERA == "D435" :
    DEPTH_CAMERA_RES = 640,480
    COLOR_CAMERA_RES = 640,480
elif REALSENSE_CAMERA == "L515" :
    DEPTH_CAMERA_RES = 1024,768
    COLOR_CAMERA_RES = 1280,720

## Setting for Pupil_tracker
addr = '127.0.0.1' # remote ip or localhost
req_port = "50020" # same as in the pupil remote gui

## Setting for blurring image
pupil_diameter = 2e-3
eye_length = 24e-3
eye_relief = 1e-1
kernel_radius_pixel = 21
#res_window = 21,21
#window_size = 0.2e-3, 0.2e-3
num_color_img_list = 8

# Convert matrix default setting
convert_matrix = np.array([[1.0078, 0.1722, 0.0502], [0, 0, 0], [0.0532, -0.6341, 0.7817]])


def make_convert_matrix(sub):
    """
    front side = input degree is 184
    (x_1, y_1, z_1): world coordinate
    (x_2, y_2, z_2): viewing coordinate (pupil camera coordinate)
    """
    coord_1 = None
    coord_2 = None

    degree = input("type observed degree:")
    while(degree != 'end' and degree != '-1'):
        degree = float(degree)
        theta_1 = pi / 2
        phi_1 = (degree - 184 + 90) * pi / 180
        x_1 = - np.sin(theta_1) * np.cos(phi_1)
        y_1 = np.cos(theta_1)
        z_1 = np.sin(theta_1) * np.sin(phi_1)

        sub.connect(b"tcp://%s:%s" %(addr.encode('utf-8'),sub_port))
        topic,msg_1 =  sub.recv_multipart()
        message_1 = loads(msg_1)
        theta_2 = message_1[b'theta']
        phi_2 = message_1[b'phi']
        sub.disconnect(b"tcp://%s:%s" %(addr.encode('utf-8'),sub_port))

        x_2 = np.sin(theta_2) * np.cos(phi_2)
        y_2 = np.cos(theta_2)
        z_2 = np.sin(theta_2) * np.sin(phi_2)

        if coord_1 is None:
            coord_1 = np.array([[x_1, y_1, z_1]])
        else:
            coord_1 = np.append(coord_1, [[x_1, y_1, z_1]], axis=0)
        if coord_2 is None:
            coord_2 = np.array([[x_2, y_2, z_2]])
        else:
            coord_2 = np.append(coord_2, [[x_2, y_2, z_2]], axis=0)

        degree = input("type observed degree:")

    # coord_1 = np.array([[-0.5,	0,	0.866025404],[-0.342020143,	0,	0.939692621],[-0.173648178,	0,	0.984807753],[0,	0,	1],[0.173648178,	0,	0.984807753],[0.342020143,	0,	0.939692621],[0.5,	0,	0.866025404]])
    # coord_2 = np.array([[-0.420787921,	-0.390899842,	0.818617639],[-0.32087076,	-0.557559388,	0.765617061],[-0.137999648,	-0.649991026,	0.747307007],[0.083080865,	-0.662271994,	0.74464312],[0.274478632,	-0.610046409,	0.743306706],[0.418590941,	-0.480357372,	0.770738879],[0.460415755,	-0.314987365,	0.829939933]])

    print("coord_1 : ", coord_1)
    print("coord_2 : ", coord_2)

    model_x = LinearRegression(fit_intercept=False).fit(coord_2,coord_1[:,0])
    model_y = LinearRegression(fit_intercept=False).fit(coord_2, coord_1[:,1])
    model_z = LinearRegression(fit_intercept=False).fit(coord_2, coord_1[:,2])

    convert_matrix = np.array([model_x.coef_, model_y.coef_, model_z.coef_])
    return convert_matrix


def convert_pupil_to_realsense(theta, phi) :
    x = np.sin(theta) * np.cos(phi)
    y = np.cos(theta)
    z = np.sin(theta) * np.sin(phi)
    coord = np.array([[x],[y],[z]])

    converted_coord = np.dot(convert_matrix, coord)

    converted_x, converted_y, converted_z = converted_coord
    converted_theta = np.arctan(converted_y / converted_z)
    converted_phi = np.arctan(converted_x / converted_z)

    return converted_theta, converted_phi


def inverse_filtering(color_img, depth_img, gaze_depth) :
    eye_focal_length = 1 / (1 / gaze_depth + 1 / eye_length)
    c = 1e+4   #coefficient for gaussian psf
    color_img_list = []
    color_img = color_img.astype(float)
    depth_img = depth_img.astype(float)
    blurred_image = np.zeros_like(color_img)

    # Calculate target intensity sum
    target_intensity_sum = np.sum(color_img)
    
    RES = COLOR_CAMERA_RES  #resolution of color_img, detph_img
    x,y = np.meshgrid(np.linspace(-RES[0]//2, RES[0]//2 - 1, RES[0]), np.linspace(-RES[1]//2, RES[1]//2 - 1, RES[1]))
    radius = np.sqrt(x*x + y*y)

    depth_list = depth_img[depth_img > 0]

    percentiles = np.linspace(0,100,num_color_img_list+1)
    depth_bound_list = np.percentile(depth_list, percentiles, interpolation='nearest')

    depth_list = np.unique(depth_list)

    for idx in range(num_color_img_list) :
        pixel_select = np.zeros_like(depth_img)
        for depth in depth_list :
            if depth_bound_list[idx] <= depth and depth < depth_bound_list[idx+1] :
                pixel_select[depth_img == depth] = 1
        if idx == num_color_img_list - 1 :
            pixel_select[depth_img == depth_bound_list[idx+1]] = 1

        depth_select_list = depth_img[pixel_select == 1]
        if len(depth_select_list) == 0 :
            continue
        depth_idx = np.mean(depth_select_list)
        #if int(gaze_depth * 1000) in depth_select_list :
        #    depth_idx = int(gaze_depth * 1000)

        pixel_select = np.stack((pixel_select, pixel_select, pixel_select), axis = 2)
        color_img_idx = color_img * pixel_select

        color_img_list.append((color_img_idx, depth_idx / 1000.0))

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

        compensate_R = restoration.wiener(R_img_idx, kernel, 1e+0, clip=False)
        compensate_G = restoration.wiener(G_img_idx, kernel, 1e+0, clip=False)
        compensate_B = restoration.wiener(B_img_idx, kernel, 1e+0, clip=False)

        compensate_img = np.stack((compensate_R, compensate_G, compensate_B), axis = 2)
        blurred_image += compensate_img

        #print("depth idx : ", depth_idx)
        #print("kernel max :", kernel.max())

    pixel_select = np.zeros_like(depth_img)
    pixel_select[depth_img == 0] = 1
    pixel_select = np.stack((pixel_select, pixel_select, pixel_select), axis = 2)
    color_img_zero_depth = color_img * pixel_select
    blurred_image += color_img_zero_depth  #just add zero depth pixel to blurred_image

    blurred_image = np.clip(blurred_image,0,np.max(blurred_image))
    blurred_image = blurred_image / np.sum(blurred_image) * target_intensity_sum / 255.0
    blurred_image = np.clip(blurred_image,0,1)

    return blurred_image


if __name__ == "__main__":

    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.depth, DEPTH_CAMERA_RES[0], DEPTH_CAMERA_RES[1], rs.format.z16, 30)
    config.enable_stream(rs.stream.color, COLOR_CAMERA_RES[0], COLOR_CAMERA_RES[1], rs.format.bgr8, 30)

    # Align process for realsense frames
    align_to = rs.stream.color
    align = rs.align(align_to)

    # Start streaming for Realsense
    pipeline.start(config)

    # Start connecting pupil tracker
    context = zmq.Context()
    req = context.socket(zmq.REQ)   #open a req port to talk to pupil
    req.connect("tcp://%s:%s" %(addr,req_port))
    req.send(b'SUB_PORT')   # ask for the sub port
    sub_port = req.recv()

    # open a sub port to listen to pupil in eye_1_3d
    sub_1_3d = context.socket(zmq.SUB)
    sub_1_3d.connect(b"tcp://%s:%s" %(addr.encode('utf-8'),sub_port))
    sub_1_3d.setsockopt(zmq.SUBSCRIBE, b'pupil.1.3d')

    need_calculate = input("Start Calculating convert matrix?(Y/n) : ")

    if (need_calculate.upper() == "Y") :
        convert_matrix = make_convert_matrix(sub_1_3d)
    print("convert matrix : ", convert_matrix)
    np.save('./convert_matrix',convert_matrix)

    input("Start convert blurred image(press enter)")
    time_0 = time.time()

    # Start convert blurred image
    try:
        while True:
            current_time = time.time()

            # Collect Data from pupil_tracker &  Wait for a coherent pair of frames: depth and color
            sub_1_3d.connect(b"tcp://%s:%s" %(addr.encode('utf-8'),sub_port))
            topic,msg_1 =  sub_1_3d.recv_multipart() # pupil tracker (maximum 120Hz)

            frames = pipeline.wait_for_frames() # realsense (maximum 30Hz)

            aligned_frames = align.process(frames)

            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            message_1 = loads(msg_1)
            theta = message_1[b'theta']
            phi = message_1[b'phi']

            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            # Check tracking point on images
            converted_theta, converted_phi = convert_pupil_to_realsense(theta, phi)

            H, W = color_image.shape[0], color_image.shape[1]
            point_y = int(H/2 + H/2 * (np.tan(converted_theta) / np.tan(COLOR_CAMERA_MAX_THETA)))
            point_x = int(W/2 + W/2 * (np.tan(converted_phi) / np.tan(COLOR_CAMERA_MAX_PHI)))
            point_y = np.clip(point_y, 0, H-1)
            point_x = np.clip(point_x, 0, W-1)

            depth_colormap = cv2.circle(depth_colormap, (point_x, point_y), 3, (0,255,0), -1)
            text = "depth : " + str(depth_image[point_y][point_x]) + "mm"
            depth_colormap = cv2.putText(depth_colormap, text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            color_image = cv2.circle(color_image, (point_x, point_y), 3, (0,255,0), -1)

            blurred_image = inverse_filtering(color_image, depth_image, depth_image[point_y][point_x] / 1000.0)
            blurred_image = cv2.circle(blurred_image, (point_x, point_y), 3, (0,255,0), -1)

            print("time : ", round(current_time - time_0, 4))
            print("theta, phi : ", theta, phi)
            print("position(x,y), depth : ", point_x, point_y, depth_image[point_y][point_x], "\n")


            # Show images
            cv2.namedWindow('Convert_blurred_image', cv2.WND_PROP_FULLSCREEN)
            cv2.resizeWindow("Convert_blurred_image", resolution[0], resolution[1])
            cv2.moveWindow('Convert_blurred_image', screen.x - 1, screen.y - 1)
            cv2.setWindowProperty('Convert_blurred_image', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

            display_blurred_image = cv2.copyMakeBorder( blurred_image, int((resolution[1]-blurred_image.shape[0])/2), int((resolution[1]-blurred_image.shape[0])/2), int((resolution[0]-blurred_image.shape[1])/2), int((resolution[0]-blurred_image.shape[1])/2), 0)

            cv2.imshow('Convert_blurred_image', display_blurred_image)

            cv2.namedWindow('original_image', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('original_image', color_image)
            cv2.waitKey(1)

            sub_1_3d.disconnect(b"tcp://%s:%s" %(addr.encode('utf-8'),sub_port))

    finally:
        # Stop streaming
        pipeline.stop()
        sub_1_3d.disconnect(b"tcp://%s:%s" %(addr.encode('utf-8'),sub_port))