## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
##      Open CV and Numpy integration        ##
###############################################

import pyrealsense2 as rs
import numpy as np
import cv2
from math import pi
import time

## Setting for realsense
# Camera setting and tracking setting
DEPTH_CAMERA_MAX_THETA = 57 / 2.0 * (pi / 180)
DEPTH_CAMERA_MAX_PHI = 86 / 2.0 * (pi / 180)
COLOR_CAMERA_MAX_THETA = 41 / 2.0 * (pi / 180)
COLOR_CAMERA_MAX_PHI = 64 / 2.0 * (pi / 180)
converted_theta = 0 * (pi / 180)
converted_phi = 10 * (pi / 180)

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Color Config
red_color = (0,0,255)

# Start streaming
pipeline.start(config)
current_time_0 = time.time()



## Setting for Pupil_tracker
import zmq
from msgpack import loads
import threading

context = zmq.Context()

#open a req port to talk to pupil
addr = '127.0.0.1' # remote ip or localhost
req_port = "50020" # same as in the pupil remote gui
req = context.socket(zmq.REQ)
req.connect("tcp://%s:%s" %(addr,req_port))

# ask for the sub port
req.send(b'SUB_PORT')
sub_port = req.recv()

# open a sub port to listen to pupil

sub = context.socket(zmq.SUB)
sub.connect(b"tcp://%s:%s" %(addr.encode('utf-8'),sub_port))
sub.setsockopt(zmq.SUBSCRIBE, b'pupil.')

# open a sub port to listen to pupil in eye_1_3d

sub_1_3d = context.socket(zmq.SUB)
sub_1_3d.connect(b"tcp://%s:%s" %(addr.encode('utf-8'),sub_port))
sub_1_3d.setsockopt(zmq.SUBSCRIBE, b'pupil.1.3d')


import numpy as np
import math
import time
import scipy.io
from sklearn.linear_model import LinearRegression

sub.connect(b"tcp://%s:%s" %(addr.encode('utf-8'),sub_port))
sub_1_3d.connect(b"tcp://%s:%s" %(addr.encode('utf-8'),sub_port))
topic,msg_1 =  sub_1_3d.recv_multipart()
message_1 = loads(msg_1)
time0=message_1[b'timestamp']

# Convert Config

convert_matrix = np.array([[1.0078, 0.1722, 0.0502], [0, 0, 0], [0.0532, -0.6341, 0.7817]])
np.save('./convert_matrix',convert_matrix)

def make_convert_matrix():
    """
    각도기->148도가 정면이고 1방향 회전만 할 수 있게 처리해 놨으니까 하드코딩해 둠
    (x_1, y_1, z_1): world coordinate와 유사 (realsense 좌표계와 평행하지만 안구 중심이 원점임)
    (x_2, y_2, z_2): viewing coordinate (pupil coordinate)
    앞으로 pupil coordinate에서 
    """
    coord_1 = np.array()
    X_2 = np.array()
    Y_2 = np.array()
    Z_2 = np.array()

    degree = input("type observed degree: ")
    while(degree != 'end'):
        degree = eval(degree)
        theta_1 = pi/2
        phi_1 = (degree - 184 + 90)*pi/180
        x_1 = -np.sin(theta_1) * np.cos(phi_1)
        y_1 = np.cos(theta_1)
        z_1 = np.sin(theta_1) * np.sin(phi_1)

        current_time = time.time()
        topic,msg_1 =  sub_1_3d.recv_multipart()
        message_1 = loads(msg_1)
        theta_2 = message_1[b'theta']
        phi_2 = message_1[b'phi']

        x_2 = np.sin(theta_2) * np.cos(phi_2) #이게맞나..
        y_2 = np.cos(theta_2)
        z_2 = np.sin(theta_2) * np.sin(phi_2)

        coord_1 = np.append(coord_1, [[x_1, y_1, z_1]], axis=0)
        X_2 = np.append(X_2, x_2)
        Y_2 = np.append(Y_2, y_2)
        Z_2 = np.append(Z_2, z_2)

    model_x = LinearRegression().fit(coord_1, X_2)
    model_y = LinearRegression().fit(coord_1, Y_2)
    model_z = LinearRegression().fit(coord_1, Z_2)

    convert_matrix = np.array([model_x.coef_, model_y.coef_, model_z.coef_])
    np.save('./convert_matrix',convert_matrix)        



def convert_pupil_to_realsense(theta, phi) :
    x = np.sin(theta) * np.cos(phi)
    y = np.cos(theta)
    z = np.sin(theta) * np.sin(phi)
    coord = np.array([[x],[y],[z]])

    converted_coord = np.dot(convert_matrix, coord)
    print(converted_coord)

    converted_x, converted_y, converted_z = converted_coord
    converted_theta = np.arctan(converted_y / converted_z)
    converted_phi = np.arctan(converted_x / converted_z)

    print(converted_theta, converted_phi)
    return converted_theta, converted_phi


# Start Pupil_to_Realsense

try:
    while True:
        current_time = time.time()
        # Collect Data from pupil_tracker &  Wait for a coherent pair of frames: depth and color
        topic,msg_1 =  sub_1_3d.recv_multipart()
        topic,msg_1 =  sub_1_3d.recv_multipart()
        topic,msg_1 =  sub_1_3d.recv_multipart() # collect only 1 over 4 datas (120Hz -> 30Hz)
        topic,msg_1 =  sub_1_3d.recv_multipart()
        frames = pipeline.wait_for_frames() # realsense (30Hz)

        message_1 = loads(msg_1)
        time1 = message_1[b'timestamp'] - time0  
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

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.waitKey(1)


finally:
    # Stop streaming
    pipeline.stop()