import numpy as np
import random


par_names = ['tfast', 'tslow','rfast', 'rslow']

def IsG(t, par):
    amp, tfast, tslow = par
    tslow = max(tfast*1.1, tslow)
    Islow = (np.exp(-t/tslow))
    Ifast = (1-np.exp(-t/tfast))
    # IsG = Ifast
    IsG = amp*(Islow + Ifast-1)
    return np.where(t < 0, 0, IsG) #IsG if t>0 else 0 

ADC_bits = 12
ADC_smpls = 128
# ADC_smpls = 80
# ADC_smpls = 248

# ADC_bits = 8
# ADC_smpls = 32

ADC_MAX  = 2**ADC_bits - 1
ADC_ZERO = 2**(ADC_bits-1) - 0
ADC_MIN = 0

trace0 = [ int(ADC_ZERO) ]* ADC_smpls


def SiPM_wf():
    chrg = random.uniform(0.1,1)
    amp =  chrg * (ADC_MAX-ADC_ZERO) * 1.3
    par = [amp,4,14*chrg*random.uniform(1.0,1.4)]
    Dx = np.linspace(-1,ADC_smpls-1,500)
    Dy = IsG(Dx,par) + ADC_ZERO
    Dx = Dx - Dx[0]
    return Dx, Dy

def SiPM_ADC():
    chrg = random.uniform(0.1,1)
    amp =  chrg * (ADC_MAX-ADC_ZERO) * 1.3
    par = [amp,4,14*chrg*random.uniform(1.0,1.4)]
    Dx = np.linspace(-1,ADC_smpls-2,ADC_smpls)
    Dy = IsG(Dx,par) + ADC_ZERO
    Dx = Dx - Dx[0]
    Dy = np.digitize(Dy, np.arange(1,ADC_MAX+1))
    return Dx, Dy


def uint12_to_therm(values, num_bins = 16):
    values = np.asarray(values, dtype=np.uint16)
    thresholds = np.arange(0,2**11,2**11/num_bins) # 2**11+1 for endpoint
    thermometer = (values[:, None] > thresholds).astype(np.uint8)
    return thermometer#, thresholds, values

def SiPM_Therm():
    x,y = SiPM_ADC()
    return np.ravel( uint12_to_therm( y + 128 - ADC_ZERO ) )


def Nois(t, par):
    amp, tfast = par
    Ifast = (np.cos(-t/tfast))
    v = amp*(Ifast)
    return np.where(t < 0, 0, v) 


def Nois_ADC():
    chrg = random.uniform(0.8,1)
    amp =  chrg * (ADC_MAX*0.01)
    par = [amp,0.001*chrg*random.uniform(1.0,1.4)]
    Dx = np.linspace(-1,ADC_smpls-2,ADC_smpls)
    Dy = Nois(Dx,par) + ADC_ZERO
    Dx = Dx - Dx[0]
    Dy = np.digitize(Dy, np.arange(1,ADC_MAX+1))
    return Dx, Dy


def Nois_Therm():
    x,y = Nois_ADC()
    return np.ravel( uint12_to_therm( y + 128 - ADC_ZERO ) )

# print ( 1 if random.random() > 0.2 else 0 );
def skw():
    return 1 if random.random() > 0.2 else 0

##########################
# Debug #
##########################

# def SiPM_ADC(): # Debug
#     Dx = np.linspace(-1,ADC_smpls-2,ADC_smpls)
#     Dy = np.zeros_like(Dx)
#     Dy[Dx >= 64] = ADC_ZERO
#     Dy[Dx < 64] = ADC_MAX
#     # Dx = Dx - Dx[0]
#     Dy = np.digitize(Dy, np.arange(1,ADC_MAX+1))
#     return Dx, Dy

# def Nois_ADC(): # Debug
#     Dx = np.linspace(-1,ADC_smpls-2,ADC_smpls)
#     Dy = np.zeros_like(Dx)
#     Dy[Dx >= 64] = ADC_MAX
#     Dy[Dx < 64] = ADC_ZERO
#     # Dx = Dx - Dx[0]
#     Dy = np.digitize(Dy, np.arange(1,ADC_MAX+1))
#     return Dx, Dy

# def SiPM_Therm(): # Debug
#     return np.array([0,1])

# def Nois_Therm(): # Debug
#     return np.array([1,0])


# print( list(SiPM_Therm()) )