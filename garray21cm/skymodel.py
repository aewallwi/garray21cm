import numpy as np
import healpy as hp
from .data import DATA_PATH
import os


def initialize_eor(frequencies, nside_sky=256):
    """Generate EoR sky-cube.

    Parameters
    ----------
    frequencies: array-like
        1d array of frequencies (float)
    nside_sky: int, optional
        nsides of healpix sky-model
        default is 256

    Returns
    -------
    eorcube: array-like
        (npix, nfreqs) array of healpix values with arbitrary units.
    """
    eorcube = np.random.randn(len(frequencies), hp.nside2npix(nside_sky))
    eorcube -= eorcube.min()
    return eorcube


def initialize_gsm(
    frequencies,
    nside_sky=256,
    save_cube=False,
    output_dir="./",
    clobber=False,
    achromatic=False,
    achromatic_chan=None,
):
    """Initialize GSM.

    Parameters
    ----------
    frequencies: array-like
        1d-array of frequencies (float)
    nside_sky: int
        nsides of healpix sky-model
    save_cube: bool, optional
        if True, save data to a numpy array to save time.
    output_dir: str, optional
    achromatic: bool, optional
        if True, then set entire sky cube equal to slice
        defined by achromatic freq
        default is False.
    achromatic_chan: int
        index of freq in frequencies to copy for achromatic sky.
        default is None-> use middle frequency.
    Returns
    -------
    gsmcube: array-like
        (npix, nfreqs) array of healpix values in Jy / sr.
    """
    if achromatic and achromatic_chan is None:
        achromatic_chan = len(frequencies) // 2

    gsm_file = os.path.join(
        output_dir,
        f"gsm_cube_f0_{frequencies[0]/1e6:.1f}MHz_nf_{len(frequencies)}_df_{np.mean(np.diff(frequencies/1e3)):.1f}_kHz_nside_{nside_sky}.npz",
    )
    if not os.path.exists(gsm_file) or clobber:
        from pygdsm import GlobalSkyModel

        gsm = GlobalSkyModel(freq_unit="Hz")
        rot = hp.rotator.Rotator(coord=["G", "C"])
        if achromatic:
            mapslice = gsm.generate(frequencies[achromatic_chan])
            mapslice = hp.ud_grade(mapslice, nside_sky)
            mapslice = rot.rotate_map_pixel(mapslice)
            gsmcube = np.asarray([mapslice for f in frequencies])
        else:
            gsmcube = np.zeros((len(frequencies), hp.nside2npix(nside_sky)))
            for fnum, f in enumerate(frequencies):
                mapslice = gsm.generate(f)
                mapslice = hp.ud_grade(mapslice, nside_sky)
                # convert from galactic to celestial
                gsmcube[fnum] = rot.rotate_map_pixel(mapslice)
        # convert gsm cube from K to Jy / Sr. multiplying by 2 k_b / lambda^2 * ([Joules / meter^2 / Jy] =1e26)
        gsmcube = 2 * gsmcube * 1.4e-23 / 1e-26 / (3e8 / frequencies[:, None]) ** 2
        np.savez(gsm_file, map=gsmcube)
    else:
        gsmcube = np.load(gsm_file)["map"]
    return gsmcube


def add_gleam(frequencies, hp_input, nsrcs=10000, achromatic=False, achromatic_chan=None):
    """Add GLEAM sources to a map via nearest neighbor gridding.

    Parameters
    ----------
    frequencies: array-like
        1d-array of frequencies (float)
    hp_input: array-like
        Nfreqs x Npix healpix array (units of Jy / Sr) to add gleam sources to.

    Returns
    -------
    hp_input: array-like
        hp_input array with gleam sources added in.
    """
    if achromatic and achromatic_chan is None:
        achromatic_chan = len(frequencies) // 2
    npix = hp_input.shape[1]
    nside = hp.npix2nside(npix)
    pixarea = hp.nside2pixarea(nside)
    theta, phi = hp.pix2ang(nside, range(npix))
    gleam_srcs = np.loadtxt(os.path.join(DATA_PATH, "catalogs/gleam_bright.txt"), skiprows=44)[:, :nsrcs]
    if achromatic:
        freq_array = np.asarray([frequencies[achromatic_chan] for f in frequencies])
    else:
        freq_array = frequencies
    for srcrow in gleam_srcs:
        ra = np.radians(srcrow[0])
        zen = np.pi / 2 - np.radians(srcrow[1])
        f200 = srcrow[-1]
        alpha = srcrow[-2]
        pixel = hp.ang2pix(nside, zen, ra)
        hp_input[:, pixel] += f200 * (frequencies / 200e6) ** alpha / pixarea
    return hp_input
