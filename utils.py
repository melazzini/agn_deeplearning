from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Iterable
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
    x_smooth,y_smooth = reduce_energy_bins_number(x_raw,y_raw,times_=squeezin_factor)
    x_clipped, y_clipped = clip_spectrum(x_smooth,y_smooth,energy_left=energy_left,energy_right=energy_right)
    return normalize_spectrum(x_clipped,y_clipped)