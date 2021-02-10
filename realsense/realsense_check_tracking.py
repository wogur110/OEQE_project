## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
##      Open CV and Numpy integration        ##
###############################################

import pyrealsense2 as rs
import numpy as np
import cv2
from math import pi

# Camera setting and tracking setting
DEPTH_CAMERA_MAX_THETA = 57 / 2.0 * (pi / 180)
DEPTH_CAMERA_MAX_PHI = 86 / 2.0 * (pi / 180)
COLOR_CAMERA_MAX_THETA = 41 / 2.0 * (pi / 180)
COLOR_CAMERA_MAX_PHI = 64 / 2.0 * (pi / 180)
theta = 0 * (pi / 180)
phi = 10 * (pi / 180)

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Color Config
red_color = (0,0,255)

# Start streaming
pipeline.start(config)

try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
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
        H,W = color_image.shape[0], color_image.shape[1]
        point_y = int(H/2 - H/2 * (np.tan(theta) / np.tan(COLOR_CAMERA_MAX_THETA)))
        point_x = int(W/2 + W/2 * (np.tan(phi) / np.tan(COLOR_CAMERA_MAX_PHI)))
        color_image = cv2.line(color_image, (point_x, point_y), (point_x, point_y), red_color, 5)

        H,W = depth_colormap.shape[0], depth_colormap.shape[1]
        point_y = int(H/2 - H/2 * (np.tan(theta) / np.tan(DEPTH_CAMERA_MAX_THETA)))
        point_x = int(W/2 + W/2 * (np.tan(phi) / np.tan(DEPTH_CAMERA_MAX_PHI)))
        depth_colormap = cv2.line(depth_colormap, (point_x, point_y), (point_x, point_y), red_color, 5)
        print(point_x, point_y, depth_image[point_y][point_x])

        # Stack both images horizontally
        images = np.hstack((color_image, depth_colormap))

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)
        cv2.waitKey(1)


finally:

    # Stop streaming
    pipeline.stop()