# from opto import Opto
# import numpy as np
# import time

# o = Opto(port='COM11')
# o.connect()
# o.current(50.0)
# o.close(soft_close=True)

from pyopto import Opto

o = Opto("COM11")
o.mode('D') # default mode is current (D)
o.current(100)
o.mode('C') # focal controll mode
o.focal_power(5) # 5 dioptor (focal distance of 20 cm).
o.close()