## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
##      Open CV and Numpy integration        ##
###############################################
import pdb
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

## Setting for realsense
# Camera setting and tracking setting
DEPTH_CAMERA_MAX_THETA = 57 / 2.0 * (pi / 180)
DEPTH_CAMERA_MAX_PHI = 86 / 2.0 * (pi / 180)
COLOR_CAMERA_MAX_THETA = 41 / 2.0 * (pi / 180)
COLOR_CAMERA_MAX_PHI = 64 / 2.0 * (pi / 180)
converted_theta = 0 * (pi / 180)
converted_phi = 0 * (pi / 180)

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

## Setting for Pupil_tracker
addr = '127.0.0.1' # remote ip or localhost
req_port = "50020" # same as in the pupil remote gui

# Convert matrix default setting
convert_matrix = np.array([[1.0078, 0.1722, 0.0502], [0, 0, 0], [0.0532, -0.6341, 0.7817]])

# Color Config
red_color = (0,0,255)


def make_convert_matrix(sub):
    """
    front side = input degree is 184
    (x_1, y_1, z_1): similar to world coordinate 
    (x_2, y_2, z_2): viewing coordinate (pupil coordinate) 
    """
    coord_1 = None
    coord_2 = None

    degree = raw_input("type observed degree:")
    while(degree != 'end' and degree != '-1'):
        degree = float(degree)
        theta_1 = pi/2
        phi_1 = (degree - 184 + 90) * pi / 180
        x_1 = -np.sin(theta_1) * np.cos(phi_1)
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
    
    print(coord_1)
    print(coord_2)

    #pdb.set_trace()
    model_x = LinearRegression(fit_intercept=False).fit(coord_2,coord_1[:,0])
    model_y = LinearRegression(fit_intercept=False).fit(coord_2, coord_1[:,1])
    model_z = LinearRegression(fit_intercept=False).fit(coord_2, coord_1[:,2])

    return np.array([model_x.coef_, model_y.coef_, model_z.coef_])
       

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


if __name__ == "__main__":
    # Start streaming for Realsense
    pipeline.start(config)
    current_time_0 = time.time()

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

    #sub.connect(b"tcp://%s:%s" %(addr.encode('utf-8'),sub_port))
    sub_1_3d.connect(b"tcp://%s:%s" %(addr.encode('utf-8'),sub_port))
    topic,msg_1 =  sub_1_3d.recv_multipart()
    message_1 = loads(msg_1)
    time0=message_1[b'timestamp']

    need_calculate = raw_input("Start Calculating?(Y/n) : ")
    if (need_calculate.upper() == "Y") :
        convert_matrix = make_convert_matrix(sub_1_3d)    
    print(convert_matrix)
    np.save('./convert_matrix',convert_matrix)

    raw_input("Start Pupil_to_Realsense(press enter)")
    
    # Start Pupil_to_Realsense
    try:
        while True:
            current_time = time.time()

            # Collect Data from pupil_tracker &  Wait for a coherent pair of frames: depth and color
            sub_1_3d.connect(b"tcp://%s:%s" %(addr.encode('utf-8'),sub_port))
            topic,msg_1 =  sub_1_3d.recv_multipart() # pupil tracker (maximum 120Hz)
            
            frames = pipeline.wait_for_frames() # realsense (maximum 30Hz)

            message_1 = loads(msg_1)
            theta = message_1[b'theta']
            phi = message_1[b'phi']
            
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            # Check tracking point on images
            converted_theta, converted_phi = convert_pupil_to_realsense(theta, phi)

            H,W = color_image.shape[0], color_image.shape[1]
            point_y = int(H/2 + H/2 * (np.tan(converted_theta) / np.tan(COLOR_CAMERA_MAX_THETA)))
            point_x = int(W/2 + W/2 * (np.tan(converted_phi) / np.tan(COLOR_CAMERA_MAX_PHI)))
            point_y = np.clip(point_y, 0, H-1)
            point_x = np.clip(point_x, 0, W-1)  
            color_image = cv2.line(color_image, (point_x, point_y), (point_x, point_y), red_color, 5)

            H,W = depth_colormap.shape[0], depth_colormap.shape[1]
            point_y = int(H/2 + H/2 * (np.tan(converted_theta) / np.tan(DEPTH_CAMERA_MAX_THETA)))
            point_x = int(W/2 + W/2 * (np.tan(converted_phi) / np.tan(DEPTH_CAMERA_MAX_PHI)))
            point_y = np.clip(point_y, 0, H-1)
            point_x = np.clip(point_x, 0, W-1)        
            depth_colormap = cv2.line(depth_colormap, (point_x, point_y), (point_x, point_y), red_color, 5)
            text = "depth : " + str(depth_image[point_y][point_x]) + "mm"
            depth_colormap = cv2.putText(depth_colormap, text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, red_color, 2)
            print(round(current_time - current_time_0, 4), theta, phi)
            print(point_x, point_y, depth_image[point_y][point_x])

            # Stack both images horizontally
            images = np.hstack((color_image, depth_colormap))

            # Show images
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', images)
            cv2.waitKey(1)

            sub_1_3d.disconnect(b"tcp://%s:%s" %(addr.encode('utf-8'),sub_port))

    finally:
        # Stop streaming
        pipeline.stop()
        #sub.disconnect(b"tcp://%s:%s" %(addr.encode('utf-8'),sub_port))
        sub_1_3d.disconnect(b"tcp://%s:%s" %(addr.encode('utf-8'),sub_port))