import numpy as np

from arm_agent.topdown_orbbec_pick import _rigid_fit


def test_rigid_fit_recovers_metric_camera_to_base_transform():
    camera_mm = np.array([[0.,0.,500.], [50.,0.,520.], [0.,60.,510.], [35.,-20.,560.]])
    angle = np.deg2rad(25.)
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0.], [np.sin(angle), np.cos(angle), 0.], [0.,0.,1.]])
    translation = np.array([.32, -.14, .08])
    base_m = (camera_mm / 1000.) @ rotation.T + translation
    got_r, got_t, error = _rigid_fit(camera_mm, base_m)
    assert np.allclose(got_r, rotation, atol=1e-9)
    assert np.allclose(got_t, translation, atol=1e-9)
    assert error < 1e-6
