import numpy as np
import yaml
import os
from pyuvsim.simsetup import _complete_uvdata, initialize_uvdata_from_params
import itertools

golomb_dict = {
    1: [0],
    2: [0, 1],
    3: [0, 1, 3],
    4: [0, 1, 4, 6],
    5: [0, 1, 4, 9, 11],
    6: [0, 1, 4, 10, 12, 17],
    7: [0, 1, 4, 10, 18, 23, 25],
    8: [0, 1, 4, 9, 15, 22, 32, 34],
    9: [0, 1, 5, 12, 25, 27, 35, 41, 44],
    10: [0, 1, 6, 10, 23, 26, 34, 41, 53, 55],
    11: [0, 1, 4, 13, 28, 33, 47, 54, 64, 70, 72],
    12: [0, 2, 6, 24, 29, 40, 43, 55, 68, 75, 76, 85],
    13: [0, 2, 5, 25, 37, 43, 59, 70, 85, 89, 98, 99, 106],
    14: [0, 4, 6, 20, 35, 52, 59, 77, 78, 86, 89, 99, 122, 127],
    15: [0, 4, 20, 30, 57, 59, 62, 76, 100, 111, 123, 136, 144, 145, 151],
    16: [0, 1, 4, 11, 26, 32, 56, 68, 76, 115, 117, 134, 150, 163, 168, 177],
    17: [0, 5, 7, 17, 52, 56, 67, 80, 81, 100, 122, 138, 159, 165, 168, 191, 199],
    18: [0, 2, 10, 22, 53, 56, 82, 83, 89, 98, 130, 148, 153, 167, 188, 192, 205, 216],
    19: [
        0,
        1,
        6,
        25,
        32,
        72,
        100,
        108,
        120,
        130,
        153,
        169,
        187,
        190,
        204,
        231,
        233,
        242,
        246,
    ],
    20: [
        0,
        1,
        8,
        11,
        68,
        77,
        94,
        116,
        121,
        156,
        158,
        179,
        194,
        208,
        212,
        228,
        240,
        253,
        259,
        283,
    ],
    21: [
        0,
        2,
        24,
        56,
        77,
        82,
        83,
        95,
        129,
        144,
        179,
        186,
        195,
        255,
        265,
        285,
        293,
        296,
        310,
        329,
        333,
    ],
    22: [
        0,
        1,
        9,
        14,
        43,
        70,
        106,
        122,
        124,
        128,
        159,
        179,
        204,
        223,
        253,
        263,
        270,
        291,
        330,
        341,
        353,
        356,
    ],
    23: [
        0,
        3,
        7,
        17,
        61,
        66,
        91,
        99,
        114,
        159,
        171,
        199,
        200,
        226,
        235,
        246,
        277,
        316,
        329,
        348,
        350,
        366,
        372,
    ],
    24: [
        0,
        9,
        33,
        37,
        38,
        97,
        122,
        129,
        140,
        142,
        152,
        191,
        205,
        208,
        252,
        278,
        286,
        326,
        332,
        353,
        368,
        384,
        403,
        425,
    ],
    25: [
        0,
        12,
        29,
        39,
        72,
        91,
        146,
        157,
        160,
        161,
        166,
        191,
        207,
        214,
        258,
        290,
        316,
        354,
        372,
        394,
        396,
        431,
        459,
        467,
        480,
    ],
    26: [
        0,
        1,
        33,
        83,
        104,
        110,
        124,
        163,
        185,
        200,
        203,
        249,
        251,
        258,
        314,
        318,
        343,
        356,
        386,
        430,
        440,
        456,
        464,
        475,
        487,
        492,
    ],
    27: [
        0,
        3,
        15,
        41,
        66,
        95,
        97,
        106,
        142,
        152,
        220,
        221,
        225,
        242,
        295,
        330,
        338,
        354,
        382,
        388,
        402,
        415,
        486,
        504,
        523,
        546,
        553,
    ],
}


def generate_triangular_array():
    """Generate a triangular radial golomb ruler array.
    """
    return

def generate_1d_uniform_array(antenna_count, separation_unit=2.0, angle_EW=0.0, copies=1, copy_separation=2.0):
    """

    """
    antpos_x = []
    antpos_y = []
    # mirror
    for mirror in range(copies):
        antpos_x = np.hstack([antpos_x, np.arange(antenna_count) * separation_unit])
        antpos_y = np.hstack([antpos_y, np.ones(antenna_count) * copy_separation * mirror])
    # rotate
    antpos_x_t = np.cos(angle_EW) * antpos_x + np.sin(angle_EW) * antpos_y
    antpos_y_t = np.cos(angle_EW) * antpos_y - np.sin(angle_EW) * antpos_x
    antpos_x = antpos_x_t
    antpos_y = antpos_y_t
    antpos_z = np.zeros_like(antpos_x)
    return np.vstack([antpos_x, antpos_y, antpos_z]).T


def generate_1d_array(antenna_count, separation_unit=2.0, angle_EW=0.0, copies=1, copy_separation=2.0):
    """Generate a 1d golomb array (or n copies)

    Parameters
    ----------
    antenna_count: int
        number of antennas in each golomb ruler.
    separation_unit: float, optional
        smallest separation between antennas in the array (meters).
        default is 2.0
    angle_EW: float: optional
        angle between the array and EW direction
        units of radians
        default is 0.0
    copies: int, optional
        number of copies of golomb ruler, separated perp to ruler direction
    copy_separation: float, optional
        separation between rulers.

    Returns
    -------
    antenna_positions: array-like
        Nants x 3 array of ENU antenna positions.

    """
    antpos_x = []
    antpos_y = []
    # mirror
    for mirror in range(copies):
        antpos_x = np.hstack([antpos_x, np.asarray(golomb_dict[antenna_count]) * separation_unit])
        antpos_y = np.hstack([antpos_y, np.ones(antenna_count) * copy_separation * mirror])
    # rotate
    antpos_x_t = np.cos(angle_EW) * antpos_x + np.sin(angle_EW) * antpos_y
    antpos_y_t = np.cos(angle_EW) * antpos_y - np.sin(angle_EW) * antpos_x
    antpos_x = antpos_x_t
    antpos_y = antpos_y_t
    antpos_z = np.zeros_like(antpos_x)
    return np.vstack([antpos_x, antpos_y, antpos_z]).T


# methods for two dimensional arrays
class Point():
    """
    Helper class for generating antenna positions from intersections of lines.
    """
    def __init__(self, x, y, pid):
        """
        Init Point object.

        Parameters
        ----------
        x: float,
            x-position of point.
        y: float,
            y-position of point.
        pid: int
            integer identifier

        Returns
        -------
        N/A

        """
        self.x = x
        self.y = y
        self.id = pid

    def translate(self, x, y):
        """
        Translate point.

        Parameters
        ----------
        x: float
            x-translation
        y: float
            y-translation

        Returns
        -------
        N/A
        """
        self.x += x
        self.y += y
    def rotate(self, rot):
        yt = np.sin(rot) * self.x + np.cos(rot) * self.y
        xt = np.cos(rot) * self.x - np.sin(rot) * self.y
        self.y = yt
        self.x = xt


class Line():
    """
    Helper class for generating antenna positions from intersections of lines.
    """
    def __init__(self, p0, p1):
        """
        """
        self.slope = (p1.y - p0.y) / (p1.x - p0.x)
        self.y0 = p0.y - p0.x * self.slope
        self.p0 = p0
        self.p1 = p1

    def intersect(self, other, pid):
        xint = (other.y0 - self.y0) / (self.slope - other.slope)
        yint = self.slope * xint + self.y0
        return Point(xint, yint, pid)


def array_2d_intersection_method(order, min_spacing, opening_angle=np.pi / 3., rotation=0.0, chord_spacing='golomb'):
    """
    Generate a two dimensional array with a high proportion of baselines along radial lines

    Parameters
    ----------
    order: int
        number of antennas on each line.
    opening_angle: float, optional
        Opening angle of triangular array.
        default is pi / 3.
    rotation: float, optional
        rotation of array
        default is 0.0 -> form array lines drawn from vertices on x-axis.
    chord_spacing: str, optional
        if "golomb", draw golomb rulers on the two chords
        if "uniform", points at the end of each intersection line are spaced
        uniformly.
        if "both", one set of lines is spaced like a golomb ruler and the other is
        spaced uniformly.

    Returns
    -------
    antenna_positions: np.ndarray
        nant x 3 array of xyz antenna positions for 2d array with antennas on
        intersections of radial lines between two points.

    """
    phi = (np.pi - opening_angle) / 2.

    if chord_spacing == 'golomb':
        garr = np.asarray(golomb_dict[order][:])
        garr = garr / garr[-1]
        garr1 = garr
    elif chord_spacing == 'uniform':
        garr = np.linspace(0, 1, order)
        garr1 = garr
    elif chord_spacing == 'both':
        garr = np.asarray(golomb_dict[order][:])
        garr = garr / garr[-1]
        garr1 = np.linspace(0, 1, order)

    pid = 0
    p1coll = []
    p2coll = []
    # generate opposite exterior points
    for g in garr:
        p1coll.append(Point(g, 0, pid))
        pid += 1
    for g in garr1[:-1]:
        p2coll.append(Point(g, 0, pid))
        pid += 1
    # start out with golomb arrays or uniformly spaced arrays on x-axis.
    for i in range(len(p1coll)):
        p1coll[i].rotate(phi)
    for i in range(len(p2coll)):
        p2coll[i].rotate(phi)
    for i in range(len(p2coll)):
        # rotate collection 2 (second leg of triangle)
        # and translate it.
        p2coll[i].rotate(opening_angle)
        p2coll[i].translate(2 * np.cos(phi), 0.)
    for i in range(len(p1coll)):
        # rotate both chords by rotation.
        p1coll[i].rotate(rotation)
    for i in range(len(p2coll)):
        p2coll[i].rotate(rotation)

    # Generate interior points by drawing lines between each vertices and the
    # points on opposite chord and then computing all of the intersections between the
    # two collections of lines.
    l1coll = [Line(p1coll[0], p2) for p2 in p2coll[1:]]
    l2coll = [Line(p2coll[0], p1) for p1 in p1coll[1:-1]]

    # index antenna collections by first and last antenna
    antcollections = {(p1coll[0].id, p2.id): [p1coll[0].id, p2.id] for p2 in p2coll[1:]}
    for p1 in p1coll[1:]:
        antcollections[min(p2coll[0].id, p1.id), max(p2coll[0].id, p1.id)] = [p2coll[0].id, p1.id]

    antcollections[min(p1coll[0].id, p1coll[-1].id), max(p1coll[0].id, p1coll[-1].id)] = [p1.id for p1 in p1coll]
    antcollections[min(p2coll[0].id, p1coll[-1].id), max(p2coll[0].id, p1coll[-1].id)] = [p2.id for p2 in p2coll] + [p1coll[-1].id]


    pintcoll = []
    for l1 in l1coll:
        for l2 in l2coll:
            pint = l2.intersect(l1, pid)
            pintcoll.append(pint)
            ckey = min(l1.p0.id, l1.p1.id), max(l1.p0.id, l1.p1.id)
            antcollections[ckey].append(pint.id)
            ckey = min(l2.p0.id, l2.p1.id), max(l2.p0.id, l2.p1.id)
            antcollections[ckey].append(pint.id)
            pid += 1

    pcollection = pintcoll + p1coll + p2coll
    # sorte by id
    pcollection = sorted(pcollection, key=lambda p: p.id)
    nant = len(pcollection)
    assert len(pcollection) == nant
    separations = np.zeros(nant * (nant - 1) // 2)
    nbl = 0
    # Find smallest separation.
    for i, j in itertools.combinations(range(nant), 2):
        p1 = pcollection[i]
        p2 = pcollection[j]
        dx = p1.x - p2.x
        dy = p1.y - p2.y
        separations[nbl] = np.sqrt(dx * dx + dy * dy)
        nbl += 1
    scale_factor = min_spacing / np.min(separations)

    # build x/y/z np.ndarrays
    xvals = np.asarray([p.x for p in pcollection])
    yvals = np.asarray([p.y for p in pcollection])
    zvals = np.zeros_like(xvals)
    # and return them.
    antenna_positions = np.vstack([xvals, yvals, zvals]).T * scale_factor

    # get baseline groups with antenna numbers in each baseline
    # ordered so that EW always the same sign (avoid conjugation mismatch).
    blcoll = {}
    antpos_dict = {p.id: (p.x, p.y) for p in pcollection}
    for pc in antcollections:
        blcoll[pc] = []
        for bl in itertools.combinations(antcollections[pc], 2):
            if antpos_dict[bl[0]][0] >= antpos_dict[bl[1]][0]:
                blcoll[pc].append(bl)
            else:
                blcoll[pc].append(bl[::-1])

    return antenna_positions, blcoll




def initialize_telescope_yamls(
    antenna_positions,
    basename,
    output_dir="./",
    clobber=False,
    antenna_diameter=2.0,
    df=400e3,
    nf=200,
    f0=120e6,
    start_time=2459122.5835133335,  # .25108 + 8. / 24.,
    integration_time=100,
    Ntimes=1,
    polarizations=[
        -5,
    ],
    beam_type='airy'
):
    """Initialize observing yaml files for simulation.

    Parameters
    ----------
    antenna_positions: array-like
        Nants x 3 array of ENU antenna positions.
    basename: str
        identifying string for yamls
    output_dir: str, optional
        directory to write simulation config files.
        default is current dir ('./')
    clobber: bool, optional
        overwrite existing config files.
        default is False
    antenna_diameter: float, optional
        Diameter of antenna apertures (meters)
        default is 2.0
    df: float, optional
        frequency channel width (Hz)
        default is 400e3
    nf: integer, optional
        number of frequency channels to simulate
        Default is 200
    f0: float, optional
        minimum frequency to simulate (Hz)
        default is 120e6
    start_time: float, optional
        JD starting time of observations (units of days)
        default is 2459122.58351335 which corresponds roughly to
        4 hours LST at the HERA site in South Africa.
    integration_time: float, optional
        Duration of each integration (units of seconds)
        default is 100 seconds
    Ntimes: int, optional
        number of time samples.
    polarizations: list, optional
        list of polarizations
        default is [-5] ('xx')
    Returns
    -------
    obs_param_yaml_name: str
        path to obs_param yaml file that can be fed into
        pyuvsim.simsetup.initialize_uvdata_from_params()
    telescope_yaml_name: str
        path to telescope yaml file. Referenced in obs_param_yaml file.
    csv_file: str
        path to csv file containing antenna locations.
    """

    csv_name = os.path.join(output_dir, f"{basename}_antenna_layout.csv")
    telescope_yaml_name = os.path.join(output_dir, f"{basename}_telescope_defaults.yaml")

    telescope_yaml_dict = {
        "beam_paths": {i: {"type": beam_type} for i in range(len(antenna_positions))},
        "telescope_location": "(-30.721527777777847, 21.428305555555557, 1073.0000000046566)",
        "telescope_name": "HERA",
        "x_orientation": "east",
    }
    if beam_type == 'airy':
        telescope_yaml_dict["diameter"] = antenna_diameter
    obs_param_dict = {
        "freq": {
            "Nfreqs": int(nf),
            "bandwidth": float(nf * df),
            "start_freq": float(f0),
        },
        "telescope": {
            "array_layout": csv_name,
            "telescope_config_name": telescope_yaml_name,
        },
        "time": {
            "Ntimes": Ntimes,
            "duration_days": integration_time * Ntimes / (24 * 3600.0),
            "integration_time": integration_time,
            "start_time": start_time,
        },
        "polarization_array": [-5],
    }
    if not os.path.exists(telescope_yaml_name) or clobber:
        with open(telescope_yaml_name, "w") as telescope_yaml_file:
            yaml.safe_dump(telescope_yaml_dict, telescope_yaml_file)
    # write csv file.
    lines = []
    lines.append("Name\tNumber\tBeamID\tE    \tN    \tU\n")
    for i, x in enumerate(antenna_positions):
        lines.append(f"ANT{i}\t{i}\t{i}\t{x[0]:.4f}\t{x[1]:.4f}\t{x[2]:.4f}\n")
    if not os.path.exists(csv_name) or clobber:
        with open(csv_name, "w") as csv_file:
            csv_file.writelines(lines)

    obs_param_yaml_name = os.path.join(output_dir, f"{basename}.yaml")
    if not os.path.exists(obs_param_yaml_name) or clobber:
        with open(obs_param_yaml_name, "w") as obs_param_yaml:
            yaml.safe_dump(obs_param_dict, obs_param_yaml)
    return obs_param_yaml_name, telescope_yaml_name, csv_name


def initialize_uvdata(
    obs_param_yaml_name,
    output_dir="./",
    clobber=False,
    compress_by_redundancy=False,
):
    """Prepare configuration files and UVData to run simulation.

    Parameters
    ----------
    output_dir: str, optional
        directory to write simulation config files.
    clobber: bool, optional
        overwrite existing config files.
    compress_by_redundancy: bool, optional
        If True, compress by redundancy. Makes incompatible with VisCPU for now.

    Returns
    -------
    uvd: UVData object
        blank uvdata file
    beams: UVBeam list
    beam_ids: list
        list of beam ids

    """
    uvdata, beams, beam_ids = initialize_uvdata_from_params(obs_param_yaml_name)

    beam_ids = list(beam_ids.values())
    beams.set_obj_mode()
    _complete_uvdata(uvdata, inplace=True)
    if compress_by_redundancy:
        uvdata.compress_by_redundancy(tol=0.25 * 3e8 / uvdata.freq_array.max(), inplace=True)
    uvdata.x_orientation = 'east'
    return uvdata, beams, beam_ids
