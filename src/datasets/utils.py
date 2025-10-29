import numpy as np

def distill_uniform(arr, min_amp=10, sample_size=100, num_bins: int = 50):
    arr = arr[np.max(arr, axis=1) >= min_amp] # Here arr contains 8 and therefore has no elem after this
    maxima = np.max(arr, axis=1)
    bins = np.linspace(np.min(maxima), np.max(maxima), num_bins + 1)
    idx = np.digitize(maxima, bins) - 1
    weights = 1.0 / np.bincount(idx, minlength=num_bins)[idx]
    weights /= np.sum(weights)
    
    return arr[np.random.choice(len(arr), size=sample_size, p=weights)]

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

def uint12_to_redint(values: np.ndarray, adc_zero: int, adc_max: int, num_bits: int = 7) -> np.ndarray:
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
    offset = np.clip(np.asarray(values, dtype=np.uint16) + 128 - adc_zero, 0, adc_max - adc_zero)
    reduced = np.right_shift(offset, 12 - num_bits - 1)
    return reduced
