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
time0=pupil_position[b'timestamp']

# Coordinate transformation
s_num = 1000
data = np.zeros([s_num,5,3])

tt=0

time1 = 0
j1=0
j2=0
j3=0
j4=0
j5=0

while time1 <1000:
    topic,msg=sub.recv_multipart()
    pupil_position = loads(msg)
    x, y = pupil_position[b'norm_pos']
    time1 = pupil_position[b'timestamp'] - time0
    print(time1, x, y)

    if time1<=5:
        data[j1,0,:] = [time1, x, y]
        j1=j1+1
    else:
        if 5<time1<=10:
            data[j2,1,:] = [time1, x, y]
            j2=j2+1
        else:
            if 10<time1<=15:
                data[j3,2,:] = [time1, x, y]
                j3=j3+1
            else:
                if 15<time1<=20:
                    data[j4,3,:] = [time1, x, y]
                    j4=j4+1
                else:
                    if 20<time1<=25:
                        data[j5,4,:] = [time1, x, y]
                        j5=j5+1



scipy.io.savemat('c:/tmp/arraydata.mat', mdict={'data': data})



            
sub.disconnect(b"tcp://%s:%s" %(addr.encode('utf-8'),sub_port))
