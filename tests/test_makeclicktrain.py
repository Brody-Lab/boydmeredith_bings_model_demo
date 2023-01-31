
import bings_model_demo.model as bm
import numpy as np

def test_totalrate():
    total_rate_in = 40
    dur = 10000
    bups = bm.make_clicktrain(total_rate=total_rate_in, duration=dur)
    leftbups, rightbups = bups['left'], bups['right']
    nleft, nright = len(leftbups), len(rightbups)
    total_rate_out = (nleft+nright)/dur
    print(total_rate_in, total_rate_out)
    assert(abs(total_rate_in-total_rate_out)< .1)

def test_depressive_adaptation():
    # test stereo clicks
    click_ts = [0, 0, .1, .3]
    phi = .9
    tau_phi = .15
    C = bm.adapt_clicks(phi, tau_phi, click_ts)
    assert(np.allclose(C, np.array([0., 0., .4866, .8518]), atol=.0001))

    # test no stereo click
    click_ts = [0, .1, .3]
    C = bm.adapt_clicks(phi, tau_phi, click_ts)
    assert(np.allclose(C, np.array([1, .9487, .9615]), atol=.0001))

def test_no_adaptation():
    # test stereo clicks
    click_ts = [0, 0, .1, .3]
    phi = 1
    tau_phi = .15
    C = bm.adapt_clicks(phi, tau_phi, click_ts)
    assert(np.allclose(C, np.array([0., 0., 1., 1.]), atol=.0001))

    # test no stereo click
    click_ts = [0, .1, .3]
    C = bm.adapt_clicks(phi, tau_phi, click_ts)
    assert(np.allclose(C, np.array([1., 1., 1.]), atol=.0001))

def test_facilitative_adaptation():
    # test stereo clicks
    click_ts = [0, 0, .1, .3, .5, .55]
    phi = 1.2
    tau_phi = .15
    C = bm.adapt_clicks(phi, tau_phi, click_ts)
    assert(np.allclose(C, np.array([0., 0., .4866, .8903, 1.018, 1.1588]), atol=.0001))

    # test no stereo click
    click_ts = [0, .1, .3, .5, .55]
    C = bm.adapt_clicks(phi, tau_phi, click_ts)
    assert(np.allclose(C, np.array([1, 1.1027, 1.0852, 1.0797, 1.2118]), atol=.0001))