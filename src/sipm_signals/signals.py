from typing import Tuple
import numpy as np
import random


# =============================
# Global Constants
# =============================
ADC_BITS: int = 12
ADC_SAMPLES: int = 128

ADC_MAX: int  = 2 ** ADC_BITS - 1
ADC_ZERO: int = 2 ** ( ADC_BITS - 1 ) - 0 # TODO: why -0?
ADC_MIN: int = 0


# =============================
# Core Signal Functions
# =============================
def isg(t: np.ndarray, par: list[float]) -> np.ndarray:
    """ 
    Generate SiPM current signal (IsG)
    
    Parameters
    ----------
    t : np.ndarray
        Time axis values.
    par : tuple[float, float, float]
        Parameters for the signal function: (amplitude, t_fast, t_slow).

    Returns
    -------
    np.ndarray
        The computed SiPM current signal values. zero for t < 0.
    """
    amp, t_fast, t_slow = par
    t_slow = max(t_fast*1.1, t_slow) # ensure t_slow > t_fast
    i_slow = (np.exp(-t/t_slow))
    i_fast = (1-np.exp(-t/t_fast))
    signal = amp*(i_slow + i_fast-1)
    return np.where(t < 0, 0, signal) 

def nois(t: np.ndarray, par: list[float]) -> np.ndarray:
    """
    Generate noise waveform (Nois)

    Parameters
    ----------
    t : np.ndarray
        Time axis values.
    par : tuple[float, float]
        Parameters for the noise function: (amplitude, t_fast).

    Returns
    -------
    np.ndarray
        The computed noise waveform values. zero for t < 0.
    """
    amp, tfast = par
    Ifast = (np.cos(-t/tfast))
    v = amp*(Ifast)
    return np.where(t < 0, 0, v) 



# =============================
# Waveform Generators
# =============================
def sipm_wf() -> Tuple[np.ndarray, np.ndarray]:
    """Generate analog SiPM waveform with fine sampling."""
    chrg: float = random.uniform(0.1,1)
    amp: float =  chrg * (ADC_MAX-ADC_ZERO) * 1.3
    par: list[float] = [amp, 4, 14*chrg*random.uniform(1.0,1.4)]
    Dx: np.ndarray = np.linspace(-1, ADC_SAMPLES-1, 500)
    Dy: np.ndarray = isg(Dx, par) + ADC_ZERO # TODO: par should be tuple
    Dx: np.ndarray = Dx - Dx[0]
    return Dx, Dy

def sipm_adc() -> Tuple[np.ndarray, np.ndarray]:
    """Generate digitised SiPM waveform (ADC samples)."""
    chrg = random.uniform(0.1,1)
    amp =  chrg * (ADC_MAX-ADC_ZERO) * 1.3
    par = [amp,4,14*chrg*random.uniform(1.0,1.4)]
    Dx = np.linspace(-1,ADC_SAMPLES-2,ADC_SAMPLES)
    Dy = isg(Dx,par) + ADC_ZERO # TODO: par should be tuple
    Dx = Dx - Dx[0]
    Dy = np.digitize(Dy, np.arange(1,ADC_MAX+1))
    return Dx, Dy


def nois_adc() -> Tuple[np.ndarray, np.ndarray]:
    """Generate digitised noise waveform (ADC samples)."""
    chrg = random.uniform(0.8,1)
    amp =  chrg * (ADC_MAX*0.01)
    par = [amp, 0.001 * chrg * random.uniform(1.0,1.4)]
    Dx = np.linspace(-1,ADC_SAMPLES-2,ADC_SAMPLES)
    Dy = nois(Dx, par) + ADC_ZERO # TODO: par should be tuple
    Dx = Dx - Dx[0]
    Dy = np.digitize(Dy, np.arange(1,ADC_MAX+1))
    return Dx, Dy

def double_sipm_adc() -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates a synthetic SiPM ADC signal by combining two signals with a random offset.

    Returns:
        Tuple[np.ndarray, np.ndarray]: 
            - x: The time or sample indices of the ADC signal.
            - y: The combined ADC signal values after adding a second signal with a random offset and adjusting by ADC_ZERO.

    Notes:
        - The function calls `sipm_adc()` twice to obtain two signals.
        - A random integer offset between 5 and 30 is chosen to determine where the second signal is added to the first.
        - The second signal is added to the first signal starting from the random offset, and adjusted by subtracting `ADC_ZERO`.
        - Assumes `sipm_adc` and `ADC_ZERO` are defined elsewhere in the module.
    """
    x, y = sipm_adc()
    x_new = np.random.randint(5, 30)
    y_new = sipm_adc()[1]
    y[x_new:] += y_new[:len(x)-x_new] - ADC_ZERO
    return x, y


# ===============================
# Debug: 2-Bit
# ===============================
def sipm_adc_2bit(): # Debug
    Dx = np.linspace(-1, ADC_SAMPLES - 2, ADC_SAMPLES)
    Dy = np.zeros_like(Dx)
    Dy[Dx >= 64] = ADC_ZERO
    Dy[Dx < 64] = ADC_MAX
    Dy = np.digitize(Dy, np.arange(1, ADC_MAX + 1))
    return Dx, Dy

def nois_adc_2bit(): # Debug
    Dx = np.linspace(-1, ADC_SAMPLES - 2, ADC_SAMPLES)
    Dy = np.zeros_like(Dx)
    Dy[Dx >= 64] = ADC_MAX
    Dy[Dx < 64] = ADC_ZERO
    # Dx = Dx - Dx[0]
    Dy = np.digitize(Dy, np.arange(1,ADC_MAX+1))
    return Dx, Dy

def sipm_therm_2bit(): # Debug
    return np.array([0,1])

def nois_therm_2bit(): # Debug
    return np.array([1,0])


# =============================
# Encoding Helpers
# =============================
def uint12_to_therm(values: np.ndarray, num_bins: int = 16) -> np.ndarray:
    """
    Convert uint12 values to thermometer encoding.

    Parameters
    ----------
    values : np.ndarray
        Input ADC values.
    num_bins : int, optional
        Number of thermometer bins.

    Returns
    -------
    np.ndarray
        Thermometer-coded array of shape (len(values), num_bins).
    """
    values = np.asarray(values, dtype=np.uint16)
    thresholds = np.arange(0,2**11,2**11/num_bins) # 2**11+1 for endpoint
    thermometer = (values[:, None] > thresholds).astype(np.uint8)
    return thermometer

def uint12_to_redint(values: np.ndarray, num_bits: int = 7) -> np.ndarray:
    """
    Convert 12-bit unsigned integer ADC values to a reduced integer representation with fewer bits.

    Parameters
    ----------
    values : np.ndarray
        Input array of 12-bit unsigned integer ADC values.
    num_bits : int, optional
        Number of bits for the reduced integer representation (default is 7).

    Returns
    -------
    np.ndarray
        Array of reduced integer values with the specified number of bits.
    """
    offset = np.clip(np.asarray(values, dtype=np.int16) + 128 - ADC_ZERO, 0, ADC_MAX - ADC_ZERO)
    reduced = np.right_shift(offset, 12 - num_bits - 1)
    return reduced


def sipm_therm() -> np.ndarray:
    """Generate thermometer encoded SiPM waveform."""
    x,y = sipm_adc()

    return np.ravel( uint12_to_therm( y + 128 - ADC_ZERO ) )

def nois_therm() -> np.ndarray:
    """Generate thermometer encoded noise waveform."""
    x,y = nois_adc()
    return np.ravel( uint12_to_therm( y + 128 - ADC_ZERO ) )



# ================================================
# Generic input simulation functions for 2 classes
# ================================================

# TODO:
# sipm_inp() and nois_inp() are generic
# sipm label and nois label

SIPM_LABEL = "Good"
NOIS_LABEL = "Ugly" # "Double"

def sipm_inp():
    return uint12_to_redint( sipm_adc()[1] )

def nois_inp():
    return uint12_to_redint( double_sipm_adc()[1] )

# for constraining/validating the network
def rand_inp():
    return np.random.randint(
        low=uint12_to_redint(np.array([ADC_ZERO])),
        high=uint12_to_redint(np.array([ADC_MAX])),
        size=ADC_SAMPLES
    )

# =============================
# Data generation
# =============================

def distill_uniform(arr: np.ndarray, min_amp: int = 10, sample_size: int = 100):
    arr = arr[np.max(arr, axis=1) >= min_amp]
    maxima = np.max(arr, axis=1)
    num_bins = 50
    bins = np.linspace(np.min(maxima), np.max(maxima), num_bins + 1)
    idx = np.digitize(maxima, bins) - 1
    counts = np.bincount(idx, minlength=num_bins)
    weights = 1.0 / counts[idx]
    weights /= np.sum(weights)  # normalize
    k = sample_size
    sample_indices = np.random.choice(range(len(arr)), size=k, p=weights)

    return arr[sample_indices]


def gen_Data(good: int = 50, bad: int = 50, min_amp: int = 10, ADC_smpls: int =ADC_SAMPLES): # used to be dependent on NN[0], but = ADC_smpls
    Train_D_good = np.empty((good*20, ADC_smpls), dtype=np.uint8)
    for i in range(good*20):
        Train_D_good[i,:] = sipm_inp()
    
    Train_D_bad = np.empty((bad*20, ADC_smpls), dtype=np.uint8)
    for i in range(bad*20):
        Train_D_bad[i,:] = nois_inp()

    Train_D_good = distill_uniform(Train_D_good, min_amp = min_amp, sample_size = good)
    Train_D_bad  = distill_uniform(Train_D_bad,  min_amp = min_amp, sample_size = bad)

    return Train_D_good, Train_D_bad

# =============================
# Utility
# =============================
def skw(p: float = 0.2) -> int:
    """
    Generate a skewed binary value.

    Parameters
    ----------
    p : float, optional
        Probability of returning 0. Default is 0.2.

    Returns
    -------
    int
        1 with probability (1-p), 0 with probability p.
    """
    return 1 if random.random() > p else 0


