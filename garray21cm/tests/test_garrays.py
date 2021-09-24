import pytest
from .. import garrays
import os
import yaml
import itertools
import numpy as np


def test_generate_1d_array():
    t1 = np.vstack([2 * np.asarray([0, 1, 4, 6]), np.asarray([0.0, 0.0, 0.0, 0.0]), np.asarray([0.0, 0.0, 0.0, 0.0])]).T
    xyz1 = garrays.generate_1d_array(antenna_count=4, separation_unit=2.0, angle_EW=0.0, copies=1, copy_separation=2.0)
    assert np.allclose(xyz1, t1)


def test_initialize_telescope_yaml(tmpdir):
    tmppath = tmpdir.strpath
    antpos = garrays.generate_1d_array(
        antenna_count=4, separation_unit=2.0, angle_EW=np.pi / 3.0, copies=2, copy_separation=2.0
    )
    (
        obs_param_yaml_name,
        telescope_yaml_name,
        csv_name,
    ) = garrays.initialize_telescope_yamls(antenna_positions=antpos, basename="test", output_dir=tmppath, df=50e3)
    assert os.path.exists(obs_param_yaml_name)
    assert os.path.exists(telescope_yaml_name)
    assert os.path.exists(csv_name)

def test_array_2d_intersection_method():
    for gorder in [3, 11, 23, 5]:
        # make sure the number of points line up.
        xyz, spoke_bls = garrays.array_2d_intersection_method(order=gorder, min_spacing=5., chord_spacing='both')
        nint_points = gorder - 2
        nants = int(nint_points ** 2. + 2 * nint_points + 3)
        assert xyz.shape == (nants, 3)
        separations = np.zeros(nants * (nants - 1) // 2)
        nbl = 0
        for i, j in itertools.combinations(range(len(xyz)), 2):
            separations[nbl] = np.linalg.norm(xyz[i] - xyz[j])
            nbl += 1
        assert np.isclose(np.min(separations), 5.)
        # check that the length of spoke_bls is what we expect
        nbl_spoke = (2 + nint_points) * (1 + nint_points) * (2 + 2 * nint_points) / 2.
        # make sure that baselines only appear once
        bl_count = {}
        for blgk in spoke_bls:
            for bl in spoke_bls[blgk]:
                if bl in bl_count:
                    bl_count[bl] += 1
                elif bl[::-1] in bl_count:
                    bl_count[bl[::-1]] += 1
                else:
                    bl_count[bl] = 1

        for bl in bl_count:
            assert bl_count[bl] == 1

        assert np.sum([bl_count[bl] for bl in bl_count]) == nbl_spoke
