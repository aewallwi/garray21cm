import pytest
from .. import garrays
import os
import yaml
import numpy as np


def test_generate_1d_array():
    t1 = np.vstack([2 * np.asarray([0, 1, 4, 6]), np.asarray([0.0, 0.0, 0.0, 0.0]), np.asarray([0.0, 0.0, 0.0, 0.0])]).T
    xyz1 = garrays.generate_1d_array(
        antenna_count=4, separation_factor=2.0, angle_EW=0.0, copies=1, copy_separation=2.0
    )
    assert np.allclose(xyz1, t1)


def test_initialize_telescope_yaml(tmpdir):
    tmppath = tmpdir.strpath
    antpos = garrays.generate_1d_array(antenna_count=4, separation_factor=2.0, angle_EW=np.pi/3., copies=2, copy_separation=2.0)
    (
        obs_param_yaml_name,
        telescope_yaml_name,
        csv_name,
    ) = garrays.initialize_telescope_yamls(antenna_positions=antpos, basename='test', output_dir=tmppath, df=50e3)
    assert os.path.exists(obs_param_yaml_name)
    assert os.path.exists(telescope_yaml_name)
    assert os.path.exists(csv_name)
