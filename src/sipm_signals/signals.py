from typing import Tuple
import numpy as np
import random
from sipm_signals.input import Input

class SiPM(Input):
    def __init__(self,
                 x=None, y=None, func=None,
                 x_range: Tuple[float, float] = (0, 127),
                 n_samples: int = 128,
                 adc_bits: int = 12,
                 adc_samples: int = 128,
                 adc_min: int = 0):
        super().__init__(x, y, func, n_samples, x_range)

        self.ADC_BITS = adc_bits
        self.ADC_SAMPLES = adc_samples
        self.ADC_MIN = adc_min

        self.ADC_MAX = (2 ** self.ADC_BITS) - 1 + self.ADC_MIN
        self.ADC_ZERO = 2 ** (self.ADC_BITS - 1) - self.ADC_MIN


    # =============================
    # Core Signal Functions
    # =============================
    def isg(self, t: np.ndarray, par: list[float]) -> np.ndarray:
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

    def nois(self, t: np.ndarray, par: list[float]) -> np.ndarray:
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
    def sipm_wf(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate analog SiPM waveform with fine sampling."""
        chrg: float = random.uniform(0.1,1)
        amp: float =  chrg * (self.ADC_MAX-self.ADC_ZERO) * 1.3
        par: list[float] = [amp, 4, 14*chrg*random.uniform(1.0,1.4)]
        Dx: np.ndarray = np.linspace(-1, self.ADC_SAMPLES-1, 500)
        Dy: np.ndarray = self.isg(Dx, par) + self.ADC_ZERO # TODO: par should be tuple
        Dx: np.ndarray = Dx - Dx[0]
        return Dx, Dy

    def sipm_adc(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate digitised SiPM waveform (ADC samples)."""
        chrg = random.uniform(0.1,1)
        amp =  chrg * (self.ADC_MAX-self.ADC_ZERO) * 1.3
        par = [amp,4,14*chrg*random.uniform(1.0,1.4)]
        Dx = np.linspace(-1, self.ADC_SAMPLES-2, self.ADC_SAMPLES)
        Dy = self.isg(Dx,par) + self.ADC_ZERO # TODO: par should be tuple
        Dx = Dx - Dx[0]
        Dy = np.digitize(Dy, np.arange(1, self.ADC_MAX + 1))
        return Dx, Dy


    def nois_adc(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate digitised noise waveform (ADC samples)."""
        chrg = random.uniform(0.8,1)
        amp =  chrg * (self.ADC_MAX*0.01)
        par = [amp, 0.001 * chrg * random.uniform(1.0,1.4)]
        Dx = np.linspace(-1, self.ADC_SAMPLES-2, self.ADC_SAMPLES)
        Dy = self.nois(Dx, par) + self.ADC_ZERO # TODO: par should be tuple
        Dx = Dx - Dx[0]
        Dy = np.digitize(Dy, np.arange(1, self.ADC_MAX + 1))
        return Dx, Dy

    def double_sipm_adc(self) -> Tuple[np.ndarray, np.ndarray]:
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
        x, y = self.sipm_adc()
        x_new = np.random.randint(5, 30)
        y_new = self.sipm_adc()[1]
        y[x_new:] += y_new[:len(x)-x_new] - self.ADC_ZERO
        return x, y


    # ===============================
    # Debug: 2-Bit
    # ===============================
    def sipm_adc_2bit(self): # Debug
        Dx = np.linspace(-1, self.ADC_SAMPLES - 2, self.ADC_SAMPLES)
        Dy = np.zeros_like(Dx)
        Dy[Dx >= 64] = self.ADC_ZERO
        Dy[Dx < 64] = self.ADC_MAX
        Dy = np.digitize(Dy, np.arange(1, self.ADC_MAX + 1))
        return Dx, Dy

    def nois_adc_2bit(self): # Debug
        Dx = np.linspace(-1, self.ADC_SAMPLES - 2, self.ADC_SAMPLES)
        Dy = np.zeros_like(Dx)
        Dy[Dx >= 64] = self.ADC_MAX
        Dy[Dx < 64] = self.ADC_ZERO
        # Dx = Dx - Dx[0]
        Dy = np.digitize(Dy, np.arange(1, self.ADC_MAX + 1))
        return Dx, Dy

    def sipm_therm_2bit(self): # Debug
        return np.array([0,1])

    def nois_therm_2bit(self): # Debug
        return np.array([1,0])


    # =============================
    # Encoding Helpers
    # =============================
    def uint12_to_therm(self, values: np.ndarray, num_bins: int = 16) -> np.ndarray:
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

    def uint12_to_redint(self, values: np.ndarray, num_bits: int = 7) -> np.ndarray:
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
        offset = np.clip(np.asarray(values, dtype=np.int16) + 128 - self.ADC_ZERO, 0, self.ADC_MAX - self.ADC_ZERO)
        reduced = np.right_shift(offset, 12 - num_bits - 1)
        return reduced


    def sipm_therm(self) -> np.ndarray:
        """Generate thermometer encoded SiPM waveform."""
        x,y = self.sipm_adc()

        return np.ravel( self.uint12_to_therm( y + 128 - self.ADC_ZERO ) )

    def nois_therm(self) -> np.ndarray:
        """Generate thermometer encoded noise waveform."""
        x,y = self.nois_adc()
        return np.ravel( self.uint12_to_therm( y + 128 - self.ADC_ZERO ) )



    # ================================================
    # Generic input simulation functions for 2 classes
    # ================================================

    # TODO: maybe better to use enums
    CLASSIFICATION  = "Signal"

    SIGNAL_LABLES   = ["Good", "Ugly"] #, "Bad"]
    SIGNAL_LABLE_OTHER = "Either"
    SIGNAL_OUTCOMES = [[1,0], [0,1]]
    SIGNAL_DICT_TUPLE_TO_LABEL = dict( zip([tuple(l) for l in SIGNAL_OUTCOMES], SIGNAL_LABLES) )

    #CLASSIFICATION = "MNIST"
    #MNIST_LABLES   = [str(i) for i in range(10)]
    #MNIST_LABLE_Other = str(MNIST_LABLES) # none-of-the-above label=next spare number
    #MNIST_OUTCOMES = np.diag(np.ones(len(MNIST_LABLES), dtype=np.uint8)) # one-hot
    #MNIST_DICT_tuple_to_label = dict( zip([tuple(l) for l in MNIST_OUTCOMES], MNIST_LABLES) )


    def signal_good_inp(self):
        return self.uint12_to_redint( self.sipm_adc()[1] )

    def signal_ugly_inp(self):
        return self.uint12_to_redint( self.double_sipm_adc()[1] )

    # for constraining/validating the network
    def other_inp(self):
        return np.random.randint(
            low=self.uint12_to_redint(np.array([self.ADC_ZERO])),
            high=self.uint12_to_redint(np.array([self.ADC_MAX])),
            size=self.ADC_SAMPLES
        )

    # =============================
    # Data generation
    # =============================

    


    def gen_Data_Labled(self, n_frames: int = 50, min_amp: int = 10): # used to be dependent on NN[0], but = ADC_smpls
        Train_D_good = np.empty((n_frames*20, self.ADC_SAMPLES), dtype=np.uint8)
        for i in range(n_frames*20):
            Train_D_good[i,:] = self.signal_good_inp()
        
        Train_D_bad = np.empty((n_frames*20, self.ADC_SAMPLES), dtype=np.uint8)
        for i in range(n_frames*20):
            Train_D_bad[i,:] = self.signal_ugly_inp()

        Train_D_good = self.distill_uniform(Train_D_good, min_amp = min_amp, sample_size = n_frames)
        Train_D_bad  = self.distill_uniform(Train_D_bad,  min_amp = min_amp, sample_size = n_frames)

        return np.concatenate([
            Train_D_good, 
            Train_D_bad
        ]) , np.concatenate([
            np.tile(self.SIGNAL_OUTCOMES[0], (len(Train_D_good) , 1)) , 
            np.tile(self.SIGNAL_OUTCOMES[1], (len(Train_D_bad)  , 1))
        ])      


    


