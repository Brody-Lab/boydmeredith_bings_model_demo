
import bings_model_demo.model as bm
import numpy as np

error() # create an error just to test breaking a PR

def test_totalrate():
    total_rate_in = 40
    dur = 10000
    bups = bm.make_clicktrain(total_rate=total_rate_in, duration=dur)
    leftbups, rightbups = bups['left'], bups['right']
    nleft, nright = len(leftbups), len(rightbups)
    total_rate_out = (nleft+nright)/dur
    print(total_rate_in, total_rate_out)
    assert(abs(total_rate_in-total_rate_out)< .1)
