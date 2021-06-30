import pytest
from .. import visibilities
from .. import garrays
import os
import yaml
import numpy as np


@pytest.fixture
def fiveant_yaml(tmpdir):
    tmppath = tmpdir.strpath
    ant_pos = garrays.generate_1d_array(antenna_count=5, separation_unit=3.1, angle_EW=33.0 / 180.0 * np.pi)
    telescope_yamls = garrays.initialize_telescope_yamls(
        antenna_positions=ant_pos,
        basename="test_five_ants",
        output_dir=tmppath,
        clobber=True,
        antenna_diameter=2.0,
        nf=13,
        df=400e3,
        f0=120e6,
        Ntimes=5,
    )
    obs_yaml = os.path.join(tmppath, "test_five_ants.yaml")
    return obs_yaml


def test_compute_visibilities(fiveant_yaml, tmpdir):
    tmppath = tmpdir.strpath
    uvd_gsm, uvd_eor = visibilities.compute_visibilities(
        obs_yaml=fiveant_yaml,
        basename="test_five_ants",
        output_dir=tmppath,
        eor_fg_ratio=3.7e-2,
        nside_sky=8,
        clobber=True,
        compress_by_redundancy=False,
    )
    assert np.isclose(
        np.sqrt(np.mean(np.abs(uvd_gsm.data_array) ** 2.0)),
        np.sqrt(np.mean(np.abs(uvd_eor.data_array) ** 2.0)) * 1 / (3.7e-2),
    )
    for uvdata in [uvd_gsm, uvd_eor]:
        assert uvdata.Nfreqs == 13
        assert uvdata.freq_array.min() == 120e6
        assert uvdata.Nbls == 5 * 6 / 2 - 5
        assert uvdata.Ntimes == 5
        bl_lens = np.linalg.norm(uvdata.uvw_array, axis=1)
        gtz = bl_lens > 0.0
        assert np.isclose(np.min(bl_lens[gtz]), 3.1)
        assert np.isclose(np.max(bl_lens), 3.1 * 11)
