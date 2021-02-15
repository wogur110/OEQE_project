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


import numpy as np
import math
import time
import scipy.io

sub.connect(b"tcp://%s:%s" %(addr.encode('utf-8'),sub_port))
topic,msg =  sub.recv_multipart()
pupil_position = loads(msg)
print(f"{topic}: {pupil_position}")
time0=pupil_position[b'timestamp']

flag = 0
while flag<2:
    print(f"{topic}: {pupil_position}")
    flag = flag + 1

            
sub.disconnect(b"tcp://%s:%s" %(addr.encode('utf-8'),sub_port))
