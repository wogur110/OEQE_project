'''

a script that will replay pupil server messages to serial.



as implemented here only the pupil_norm_pos is relayed.

implementing other messages to be send as well is a matter of renaming the vaiables.




'''


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

sub.connect(b"tcp://%s:%s" %(addr.encode('utf-8'),sub_port))
sub_1_3d.connect(b"tcp://%s:%s" %(addr.encode('utf-8'),sub_port))
topic,msg_1 =  sub_1_3d.recv_multipart()
message_1 = loads(msg_1)
time0=message_1[b'timestamp']

# Coordinate transformation
data_length = 1000
data_idx = 0
data = np.zeros([data_length,5])

time1 = 0

while time1 < 1:
    topic,msg_1 =  sub_1_3d.recv_multipart()
    message_1 = loads(msg_1)
    x, y = message_1[b'norm_pos']
    time1 = message_1[b'timestamp'] - time0  
    theta = message_1[b'theta']
    phi = message_1[b'phi']
    
    #print(time1, x, y, theta, phi)
    print(time1, theta, phi)
    
    if (data_idx < data_length) :
        data[data_idx, :] = [time1, x, y, theta, phi]
        data_idx = data_idx + 1

topic,msg_1 =  sub_1_3d.recv_multipart()
message_1 = loads(msg_1)
print(message_1)
    

scipy.io.savemat('./arraydata.mat', mdict={'data': data})
            
sub.disconnect(b"tcp://%s:%s" %(addr.encode('utf-8'),sub_port))
sub_1_3d.disconnect(b"tcp://%s:%s" %(addr.encode('utf-8'),sub_port))
