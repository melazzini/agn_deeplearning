from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Iterable, Final
from abc import abstractmethod, ABC

Vector1d = List[float]
"""List of numbers or similar like a 1d-ndarray object."""

DEFAULT_EPSILON = 1E-12
"""this is the default precision that we use in our calculations"""

LENGTH = 'L'
TIME = 'T'
MASS = 'M'
TEMPERATURE = 'K'
ANGLE = 'A'
ENERGY = 'ML2T-2'
N_H_DIM = '_N_H_'


def solid_angle(half_opening_angle: float) -> float:
    """Returns the solid angle that corresponds
    to the given half-opening angle.

    For example if the HOA is pi/2 then the function
    returns 2pi as the solid angle.

    Args:
        half_opening_angle (float): 

    Returns:
        float: the corresponding solid angle
    """
    return 2*np.pi*(1.0 - np.cos(half_opening_angle))


def solid_angle_doubled(half_opening_angle: float) -> float:
    """Returns the doubled solid angle that corresponds
    to the given half-opening angle.

    Args:
        half_opening_angle (float): 

    Returns:
        float: The doubled solid angle
    """
    return 2.0 * solid_angle(half_opening_angle)


def x_y(path: str, index_left: int = 0, index_right: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Builds a pair of np ndarrays using the input file, which is in the form
    of a white-space separated columns, for example:
                    1\t   0.1  \t   0 \n
                    2\t   0.2  \t   0 \n
                    3\t   0.3  \t   1 \n
                    4\t   0.9  \t   0 \n

    The file can have more that two columns, but the length of the 
    columns must be the same!

    Args:
        path (str): the path to the input file that contains the data
        index_left (int, optional): the index of the x-column in the file. Defaults to 0 (ie the first column).
        index_right (int, optional): the index of the y-column in the file . Defaults to 1 (ie the second column).

    Returns:
        tuple(np.ndarray,np.ndarray): x, y ndarrays correspondingly
    """
    try:  # normal case of multiple lines
        data = np.loadtxt(path)
        return data[:, index_left], data[:, index_right]
    except:  # case of only one single line in the file, for example: -1 50.92 30
        data = np.loadtxt(path)
        return np.array([data[index_left]]), np.array([data[index_right]])


def x_y_z(path: str, index_left: int = 0, index_mid: int = 1, index_right: int = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Builds three np ndarrays using the input file, which is in the form
    of a white-space separated columns, for example:
                    1\t   0.1  \t   0 \n
                    2\t   0.2  \t   0 \n
                    3\t   0.3  \t   1 \n
                    4\t   0.9  \t   0 \n

    The file can have more that three columns, but the length of the 
    columns must be the same!

    Args:
        path (str): the path to the input file that contains the data
        index_left (int, optional): the index of the x-column in the file. Defaults to 0 (ie the first column).
        index_mid (int, optional): the index of the y-column in the file . Defaults to 1 (ie the second column).
        index_right (int, optional): the index of the z-column in the file . Defaults to 2 (ie the third column).

    Returns:
        tuple(np.ndarray, np.ndarray, np.ndarray): x,y,z ndarrays correspondingly
    """

    try:  # normal case of multiple lines
        data = np.loadtxt(path)
        return data[:, index_left], data[:, index_mid], data[:, index_right]
    except:  # case of only one single line in the file, for example: -1 50.92 30
        data = np.loadtxt(path)
        return np.array([data[index_left]]), np.array([data[index_mid]]), np.array([data[index_right]])


def mean_2d(x_ar: Vector1d, y_ar: Vector1d) -> Tuple[float, float]:
    """Get the mean of two 1d-Vectors at the same time.

    The length of both vectors must be same!

    Args:
        x_ar (Vector1d): x-Vector, for example [1,2,3]
        y_ar (Vector1d): y-Vector, for example [0,1,2]

    Raises:
        ValueError: if the lengths of the vectors is different.

    Returns:
        Tuple[float, float]: mean of the x-vector, mean of the y-vector
    """
    if(len(x_ar) != len(y_ar)):
        raise ValueError("the length of the arrays must be equal!")
    return np.mean(x_ar), np.mean(y_ar)


@dataclass
class ValueAndError:
    value: float
    err: float


@dataclass
class Interval2D:
    left: float
    right: float

    def __contains__(self, value: float) -> bool:
        if self.left < self.right:
            if self.left <= value <= self.right:
                return True
            else:
                return False
        else:
            if self.left >= value >= self.right:
                return True
            else:
                return False


class EnergyInterval(Interval2D):
    pass


class AngularInterval(Interval2D):
    """Represents an angular interval.
    """

    def __init__(self, beg: float, length: float):
        """Creates an instance with the staring
        angle and the length of it.

        You have to keep in mind the units that
        you are using, because this information
        is not kept in the created object.

        Args:
            beg (float): the starting angle
            length (float): the length of the interval
        """
        self.beg = beg
        self.length = length
        self.end = self.beg + self.length
        super().__init__(left=self.beg, right=self.end)

    def from_deg_to_rad(self) -> AngularInterval:
        """Translates from degrees to radians the angular interval.

        Here it's assumed that the initial units
        are degrees.

        Returns:
            AngularInterval: self
        """
        # self.beg = np.radians(self.beg)
        # self.length = np.radians(self.length)
        return AngularInterval(beg=np.radians(self.beg), length=np.radians(self.length))

    def from_rad_to_deg(self) -> AngularInterval:
        """Translates from radians to degrees the angular interval.

        Here it's assumed that the initial units
        are radians.

        Returns:
            AngularInterval: self
        """
        self.beg = np.degrees(self.beg)
        self.length = np.degrees(self.length)
        return self

    def __str__(self):
        return f'({self.beg}, {self.length})'


class UnitsPolicy(ABC):

    @abstractmethod
    def translate_energy(self, value: float) -> float:
        pass

    @abstractmethod
    def translate_length(self, value: float) -> float:
        pass

    @abstractmethod
    def translate_angle(self, value: float) -> float:
        pass


@dataclass
class Histo:
    """This class represents a histogram.

    This class holds the raw data, because this
    is usually convenient for different calculations.

    Plus it provides basic statistical methods: mean and std.
    """
    bins: np.ndarray
    counts: np.ndarray
    counts_err: np.ndarray
    raw_data: np.ndarray

    def mean(self) -> float:
        """Get the mean of the raw data described by the histogram.

        This value is calculated every time you call this method.
        Thus: 
                Cache this value for performance!

        Returns:
            float: mean value of the raw data.
        """

        return np.mean(self.raw_data)

    def std(self) -> float:
        """Get the std of the raw data described by the histogram.

        This value is calculated every time you call this method.
        Thus: 
                Cache this value for performance!

        Returns:
            float: standard deviation of the raw data.
        """
        mean_ = self.mean()
        n = len(self.raw_data)
        return np.sqrt(sum(((self.raw_data-mean_)**2)/n))/mean_


def product_transport_err(data: Iterable[ValueAndError]) -> float:
    """Get the error of a product or division.

    Args:
        data (Iterable[ValueAndError]): list of ValueAndError to calculate the total error.

    See: https://studfile.net/preview/3130553/page:4/

    Returns:
        float: error
    """
    return sum([(item.err/item.value)**2 for item in data])**0.5



def chi2(observed, expected):
    return np.sum(((observed-expected)**2)/expected)

def distance_to_function(point_x, point_y, f_x_left, f_x_right, f, steps, *args):

    x_array = np.linspace(f_x_left, f_x_right, num=steps)

    min_d = np.inf
    min_x = np.inf
    min_y = np.inf

    for x in x_array:
        y = f(x, *args)

        d = np.sqrt((x-point_x)**2 + (y-point_y)**2)

        if d < min_d:
            min_d = d
            min_x = x
            min_y = y

    return min_d, min_x, min_y


def half_energy_bins_number(x, y):
    """
    This algorithm reduces the number of energy bins by half.
    The resulting x will be built from the average values of
    the original x, and the resulting y will be the sum of
    the original y.
    """
    x_result = np.zeros(int(len(x)/2))
    y_result = np.zeros(int(len(y)/2))
    
    for i in range(len(x_result)):
        x_result[i] = (x[2*i] + x[2*i+1])/2
        y_result[i] = y[2*i] + y[2*i+1]
    return x_result,y_result

def reduce_energy_bins_number(x,y,times_:int=1):
    """
    Reduces the energy bins a number of times.
    """
    x_new,y_new = x,y
    for i in range(times_):
        x_new,y_new = half_energy_bins_number(x_new,y_new)
    return x_new,y_new


def normalize_spectrum(x,y):
    return x,(y-np.mean(y))/np.std(y)

def clip_spectrum(x,y,energy_left,energy_right):
    """
    Returns a new spectrum without the energy intervals
    that do not belong to [energy_left,energy_right]
    """
    x_new = []
    y_new = []

    for x_i, y_i in zip(x,y):
        if energy_left <= x_i <= energy_right:
            x_new+=[x_i]
            y_new+=[y_i]

    return np.array(x_new), np.array(y_new)

def prepare_spectrum(x_raw,y_raw,energy_left:float,energy_right:float,squeezin_factor:int):
    """
    It receives a given raw spectrum, then reduce the number of energy bins by squeezing it
    the given squeezing_factor number of times, then this function clips the resulting spectrum,
    and then normalized it before returning it as a tupple (x_new,y_new)
    """
    if np.sum(y_raw) < 1500:
        raise ValueError("Two few photons in the spectrum")
    
    x_smooth,y_smooth = reduce_energy_bins_number(x_raw,y_raw,times_=squeezin_factor)
    x_clipped, y_clipped = clip_spectrum(x_smooth,y_smooth,energy_left=energy_left,energy_right=energy_right)
    return normalize_spectrum(x_clipped,y_clipped)


AGN_VIEWING_DIRECTIONS_DEG: Final[Dict[str, AngularInterval]] = {
    "6075": AngularInterval(60, 15),
    "7590": AngularInterval(75, 15),
    "6070": AngularInterval(60, 10),
    "7080": AngularInterval(70, 10),
    "8090": AngularInterval(80, 10),
}


AGN_IRON_ABUNDANCE: Final[Dict[str, float]] = {
    "05xfe": 0.5,
    "0525xfe": 0.525,
    "07xfe": 0.7,
    "1xfe": 1,
    "105xfe": 1.05,
    "15xfe": 1.5,
    "2xfe": 2,
    "21xfe": 2.1,
    "4xfe": 4,
}

AGN_NH_AVERAGE: Final[Dict[str:float]] = {
    "22": 1e22,
    "222": 2e22,
    "522": 5e22,
    "822": 8e22,
    "23": 1e23,
    "223": 2e23,
    "323": 3e23,
    "523": 5e23,
    "24": 1e24,
    "224": 2e24,
    "324": 3e24,
    "424": 4e24,
    "524": 5e24,
    "824": 8e24,
    "25": 1e25,
    "225": 2e25,
}

class ColumnDensityInterval(Interval2D):
    pass

class ColumnDensityGrid:
    """
    This class defines the grid of column densities
    used to group spectrum data from agn simulations.

    Basically, you use it to get the index of the given
    column density on the whole grid for the current project.

    This can be useful to group spectrum data and build labels
    for spectrum files, according to the given values of
    the column density.

    You are not require to use any specific units, but
    probably you want to work in [N_H]=cm^{-2}.

    ===========================

    example01:

    nh_grid = ColumnDensityGrid(left_nh=1e22, right_nh=2e24, n_intervals=30)

    print(nh_grid.index(nh=1e23))

    ==========================
    """

    def __init__(self, left_nh: float, right_nh: float, n_intervals: int):
        """Initializes the Grid

        Args:
            left_nh (float): the left-most bound of the grid
            right_nh (float): the right-most bound of the grid
            n_intervals (int): the number of intervals of the grid
        """

        self.left = left_nh
        self.right = right_nh
        self.n_intervals = n_intervals

        self.bounds = []
        """the bounds of the grid
        """

        self.nh_list = []
        """the mid-values of the bounds
        """

        nh_bounds_raw = np.logspace(
            np.log10(self.left), np.log10(self.right), n_intervals+1)

        self.__setup_grid(nh_bounds_raw)
        self.d_nh = (np.log10(self.right) -
                     np.log10(self.left))/self.n_intervals

    def __setup_grid(self, nh_bounds_raw):
        for i, nh_left_val in enumerate(nh_bounds_raw[:-1]):

            self.nh_list += [10**((np.log10(nh_left_val) +
                                  np.log10(nh_bounds_raw[i+1]))/2)]

            self.bounds += [ColumnDensityInterval(
                left=nh_left_val, right=nh_bounds_raw[i+1])]

    def index(self, nh: float):
        """Get the grid index of the given column density.

        Args:
            nh (float): the column density
        """
        if(nh <= 0):
            return 0

        return int((np.log10(nh)-np.log10(self.left))/self.d_nh)

    def __str__(self):
        return f'{self.left:0.2g}:{self.right:0.2g}:{self.n_intervals}'

LEFT_NH = 10**(20.95)
RIGHT_NH = 10**(26.05)
NH_INTERVALS = 51

ENERGY_LEFT = 1e3
ENERGY_RIGHT = 50e3
SQUEEZING_FACTOR = 2

DEFAULT_NH_GRID: Final[ColumnDensityGrid] = ColumnDensityGrid(
    left_nh=LEFT_NH, right_nh=RIGHT_NH, n_intervals=NH_INTERVALS)


@dataclass
class NormalizedSpectrumInfo:
    nha: float # average column density, in $cm^{-2}$
    n: int # average number of clouds
    afe: float # iron abundance
    alpha: AngularInterval # viewnig angle interval, in rad
    nh: float # column density on the line of sight, in $cm^{-2}$
    component:str # spectrum component
    fluorescent_line: str # fluorescent line

    y:np.ndarray # prepared photon counts channels

    def nha_id(self):
        return DEFAULT_NH_GRID.index(self.nha)

    def nh_id(self):
        return DEFAULT_NH_GRID.index(self.nh)

    def alpha_id(self):
        if self.alpha==AngularInterval(60,10):
            return 0
        elif self.alpha==AngularInterval(70,10):
            return 1
        elif self.alpha==AngularInterval(80,10):
            return 2
        else:
            raise RuntimeError("the alpha value is not valid!")
    
    @staticmethod
    def build_normalized_spectrum_info(spectrum_file_path:str, energy_left:float=ENERGY_LEFT, energy_right:float=ENERGY_RIGHT, squeezing_factor:int=SQUEEZING_FACTOR):
        spectrum_filename = spectrum_file_path.split('/')[-1].split('.spectrum')[0]

        nhaver_label_str, n_str, afe_str, alpha_str, nh_num_intervals_str, nh_left_str, nh_right_str, nh_id_str, component, line = spectrum_filename.split('_')

        nha = AGN_NH_AVERAGE[nhaver_label_str]
        n = int(n_str)
        afe = AGN_IRON_ABUNDANCE[afe_str]
        alpha = AGN_VIEWING_DIRECTIONS_DEG[alpha_str]

        NormalizedSpectrumInfo.validate_nh_intervals(num=int(nh_num_intervals_str),left=float(nh_left_str),right=float(nh_right_str))
        nh = DEFAULT_NH_GRID.nh_list[int(nh_id_str)]

        x_raw, y_raw, _ = x_y_z(path=spectrum_file_path)
        _, y_prepared = prepare_spectrum(x_raw=x_raw,y_raw=y_raw,energy_left=energy_left, energy_right=energy_right, squeezin_factor=squeezing_factor)

        return NormalizedSpectrumInfo(nha=nha, n=n,afe=afe, alpha=alpha, nh=nh, component=component, fluorescent_line=line, y=y_prepared)

    @staticmethod
    def validate_nh_intervals(num:int, left:float, right:float):
        if num != NH_INTERVALS:
            raise RuntimeError("the number of column density intervals is not valid!")
        if abs(1/left-1/LEFT_NH)>DEFAULT_EPSILON:
            raise RuntimeError("the left column density interval is not valid!")
        if abs(1/right-1/RIGHT_NH)>DEFAULT_EPSILON:
            raise RuntimeError("the right column density interval is not valid!")


def print_train_time(start, end, device=None):
    """Prints difference between start and end time.

    Args:
        start (float): Start time of computation (preferred in timeit format). 
        end (float): End time of computation.
        device ([type], optional): Device that compute is running on. Defaults to None.

    Returns:
        float: time between start and end in seconds (higher is longer).
    """
    total_time = end - start
    print(f"\nTrain time on {device}: {total_time:.3f} seconds")
    return total_time


def get_train_and_test_paths(train_paths_file_path, test_paths_file_path):
    all_train_paths = []
    all_test_paths = []
    with open(train_paths_file_path) as train_paths_file:
        for line in train_paths_file:
            all_train_paths += [line]
            
    with open(test_paths_file_path) as test_paths_file:
        for line in test_paths_file:
            all_test_paths += [line]
        
    return all_train_paths, all_test_paths