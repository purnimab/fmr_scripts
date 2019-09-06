import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import convolve
from scipy.integrate import trapz
#from scipy.fftpack import fft, ifft, fftfreq

#TODO: figure out how to do background subtraction not just with H>0

def readFile(filename):
    H, dPdH, phase = np.loadtxt(filename, unpack=True, delimiter=',')
    return H, dPdH, phase

def readFileTruncatePos(filename):
    H, dPdH, phase = np.loadtxt(filename, unpack=True, delimiter=',')
    zero = np.where(H<0)[0]
    if len(zero) > 0:
        H = H[:zero[0]]
        dPdH = dPdH[:zero[0]]
        phase = phase[:zero[0]]
    return H, dPdH, phase

def Lorentzian(x, a, x0, theta, b):
    denom = b**2 + (x-x0)**2
    sym = b
    asym = x-x0
    return a/denom*(sym*np.cos(theta)+asym*np.sin(theta))

def LorentzianDerivative(x, a, x0, theta, b):
    denom = (b**2 + (x-x0)**2)**2
    sym = -2*(x-x0)*b
    asym = (b**2-(x-x0)**2)
    return a/denom*(sym*np.cos(theta)+asym*np.sin(theta))

def ExponentiallyBroaden(lor,tau):
    #exponential = np.exp(-np.arange(0,len(lor))/tau)
    exponential = np.exp(-np.linspace(0.,50*100.,num=51)/tau)
    l = len(exponential)
    #convolution = convolve(lor,exponential,mode='same')
    convolution = np.array([np.sum(lor[i:i+l]*exponential[::-1]) for i,s in enumerate(lor[l-1:])])/np.sum(exponential)
    convpart = np.array([np.sum(lor[:i+1]*exponential[:i+1][::-1])/np.sum(exponential[:i+1]) for i,s in enumerate(lor[:l-1])])
    return np.append(convpart,convolution)

def LorentzianN(x, a, x0, theta, b):
    return np.sum([Lorentzian(x,a[i],x0[i],theta[i],b[i]) for i,x0i in enumerate(x0)],axis=0)

def LorentzianDerivativeN(x, a, x0, theta, b):
    return np.sum([LorentzianDerivative(x,a[i],x0[i],theta[i],b[i]) for i,x0i in enumerate(x0)],axis=0)

def LorentzianDerivativeBroadenedN(x, a, x0, theta, b, tau):
    lor = np.sum([LorentzianDerivative(x,a[i],x0[i],theta[i],b[i]) for i,x0i in enumerate(x0)],axis=0)
    return ExponentiallyBroaden(lor,tau)

def LorentzianNWrapper(x, *args):
    a = args[0][::4]
    x0 = args[0][1::4]
    theta = args[0][2::4]
    b = args[0][3::4]
    return LorentzianN(x,a,x0,theta,b)

def LorentzianDerivativeNWrapper(x, *args):
    a = args[0][::4]
    x0 = args[0][1::4]
    theta = args[0][2::4]
    b = args[0][3::4]
    return LorentzianDerivativeN(x,a,x0,theta,b)

def LorentzianNLockInWrapperQuick(x, *args):
    tau = .03 #s - lock-in time constant for filtering
    duration = .0025 #s time to go through an integer number of samples to get all possible field points
    sampling = 256000. #Hz
    freq = 700. #Hz
    phase = 0.

    time = np.linspace(0.,duration, num=int(duration*sampling)+1) #256 kHz sampling
    ref = np.sin(time*2*np.pi*700.+phase) #700 Hz reference signal
    refmag = np.trapz(ref**2)
    sig = np.zeros_like(x)

    rms = args[0][-1]
    hac = rms*np.sqrt(2)*ref

    peakparams = args[0][:-1]

    for i,h in enumerate(x):
        field = h + hac
        signal = LorentzianNWrapper(field,peakparams)

        lix = signal*ref
        sig[i] = np.trapz(lix)/refmag
    return sig

#adapted from https://stackoverflow.com/a/52998713, answer by Jake Walden
def ewmafilter_4(data, alpha, scaling_factors=None, dtype=None, out=None, fast=True, window=128): #4 filters corresponding to 24 dB/oct. dropoff
    
    #if dtype is None:
    #    if data.dtype == np.float32:
    #        dtype = np.float32
    #    else:
    dtype = np.float64
    #else:
    #    dtype = np.dtype(dtype)
    
    interm = np.empty_like(data, dtype=dtype)
    if out is None:
        out = np.empty_like(data, dtype=dtype)
    else:
        assert out.shape == data.shape
        assert out.dtype == dtype

    if scaling_factors is None: #may be pre-computed - then takes precedence over alpha
        alpha = np.array(alpha, copy=False).astype(dtype, copy=False)
        scaling_factors = np.power(1. - alpha, np.arange(data.size + 1, dtype=dtype), dtype=dtype)
    
    #loop 1 - stores into interm
    offset = data[0]
    np.multiply(data, (alpha * scaling_factors[-2]) / scaling_factors[:-1],
                dtype=dtype, out=interm)
    np.cumsum(interm, dtype=dtype, out=interm)
    interm /= scaling_factors[-2::-1]
    if offset != 0:
        offset = np.array(offset, copy=False).astype(dtype, copy=False)
        interm += offset * scaling_factors[1:]
    
    #loop 2 - stores into out
    offset = interm[0]
    np.multiply(interm, (alpha * scaling_factors[-2]) / scaling_factors[:-1],
                dtype=dtype, out=out)
    np.cumsum(out, dtype=dtype, out=out)
    out /= scaling_factors[-2::-1]
    if offset != 0:
        offset = np.array(offset, copy=False).astype(dtype, copy=False)
        out += offset * scaling_factors[1:]

    #loop 3 - stores into interm (to save memory)
    offset = out[0]
    np.multiply(out, (alpha * scaling_factors[-2]) / scaling_factors[:-1],
                dtype=dtype, out=interm)
    np.cumsum(interm, dtype=dtype, out=interm)
    interm /= scaling_factors[-2::-1]
    if offset != 0:
        offset = np.array(offset, copy=False).astype(dtype, copy=False)
        interm += offset * scaling_factors[1:]
    
    #loop 4 - stores into out (to save memory) - final loop
    offset = interm[0]
    np.multiply(interm, (alpha * scaling_factors[-2]) / scaling_factors[:-1],
                dtype=dtype, out=out)
    np.cumsum(out, dtype=dtype, out=out)

    #TODO: can cut down on computation time by only computing the 'window'th last elements of the last ewma

    out /= scaling_factors[-2::-1]
    if offset != 0:
        offset = np.array(offset, copy=False).astype(dtype, copy=False)
        out += offset * scaling_factors[1:]

    return out

def LockIn(h, hac, peakparams, ref, alpha, window, scaling_factors=None):
    field = h + hac
    signal = LorentzianNWrapper(field, peakparams)

    lix = signal*ref
    #liy = signal*refo
    filteredx = ewmafilter_4(lix,alpha,scaling_factors=scaling_factors)
    #filteredy = ewmafilter_4(liy,alpha,scaling_factors=scaling_factors)
    #theta[i] = np.arctan2(np.mean(np.mean(filteredy[-window:])),sig[i])*180./np.pi
    return np.mean(filteredx[-window:])

def LorentzianNLockInWrapper(x, *args):
    tau = .03 #s - lock-in time constant for filtering
    duration = .1 #s 'time between measurement' in labview - may also need to record this in data file??
    sampling = 256000. #Hz
    window = 128
    freq = 700. #Hz
    phase = 0.
    dB = 24.

    time = np.linspace(0.,duration, num=int(duration*sampling)+1) #256 kHz sampling, 30ms time constant TODO: read in from file

    ref = np.sin(time*2*np.pi*700.+phase) #700 Hz reference signal
    #refo = np.cos(time*2*np.pi*700.+phase)

    alpha = 1./(1.+tau*sampling)#filter
    s = np.power(1. - np.array(alpha), np.arange(time.size + 1))

    rms = args[0][-1]
    hac = rms*np.sqrt(2)*ref
    peakparams = args[0][:-1]

    sig = [LockIn(h, hac, peakparams, ref, alpha, window, scaling_factors=s) for h in x]

    return np.array(sig)

def LorentzianDerivativeBroadenedNWrapper(x, *args):
    a = args[0][:-1:4]
    x0 = args[0][1:-1:4]
    theta = args[0][2:-1:4]
    b = args[0][3:-1:4]
    tau = args[0][-1]
    return LorentzianDerivativeBroadenedN(x,a,x0,theta,b,tau)

def guesses(H, dPdH):
    peakmax = np.argmax(dPdH)
    peakmin = np.argmin(dPdH)
    guessmag = 400.*(np.max(dPdH)-np.min(dPdH))
    guessres = (H[peakmax]+H[peakmin])/2.
    guesswidth = np.abs(H[peakmax]-H[peakmin])/2.
    return guessmag, guessres, guesswidth

def guessN(H,dPdH,N,width=1,pos='Center'):
    guessmag, guessres, guesswidth = guesses(H,dPdH)
    spacing = width*guesswidth
    if pos=='Left':
    	guessHres = np.linspace(guessres,guessres+(N-1)*spacing,N)
    elif pos=='Right':
    	guessHres = np.linspace(guessres-(N-1)*spacing,guessres,N)
    else: #pos='Center'
    	guessHres = np.linspace(guessres-(N-1)/2.*spacing,guessres+(N-1)/2.*spacing,N)
    guessN = np.concatenate(tuple(np.array([guessmag,h,0,guesswidth]) for h in guessHres))
    return guessN

def guessNWidths(H,dPdH,N,width=1,pos='Center'): #not sure if this does what it's supposed to
    guessmag, guessres, guesswidth = guesses(H,dPdH)
    spacing = width*guesswidth
    if pos=='Left':
    	guessHres = np.linspace(guessres,guessres+(N-1)*spacing,N)
    	guesswidths = np.linspace(guesswidth, N*guesswidth, N)
    elif pos=='Right':
    	guessHres = np.linspace(guessres-(N-1)*spacing,guessres,N)
    	guesswidths = np.linspace(guesswidth, N*guesswidth, N)
    else: #pos='Center'
    	guessHres = np.array([guessres]*N)
    	guesswidths = np.linspace(guesswidth, N*guesswidth, N)
    guessNw = np.concatenate(tuple(np.array([guessmag,h,0,guesswidths[i]]) for i,h in enumerate(guessHres)))
    return guessNw

def guessNplus1(oldguess,width=1,pos='Right'): #TODO: WRITE THIS
    guessmag, guessres, guesswidth = guesses(H,dPdH)
    spacing = width*guesswidth
    if pos=='Left':
    	guessHres = np.linspace(guessres,guessres+(N-1)*spacing,N)
    elif pos=='Right':
    	guessHres = np.linspace(guessres-(N-1)*spacing,guessres,N)
    else: #pos='Center'
    	guessHres = np.linspace(guessres-(N-1)/2.*spacing,guessres+(N-1)/2.*spacing,N)
    guessNplus1 = np.concatenate(tuple(np.array([guessmag,h,0,guesswidth]) for h in guessHres))
    return guessN
    
def linearBG(x, m, b):
    return m*x+b
    
def subtractBG(H, dPdH, window=6, BGfunc=linearBG):
    #need to fix this to work with negative H
    guessmag, guessres, guesswidth = guesses(H,dPdH)
    bgH = []
    bg = []
    
    upperbound = np.where(H>guessres+guesswidth*window)[0]
    if len(upperbound) > 0:
        bgend = upperbound[-1]
        bgH.extend(H[0:bgend])
        bg.extend(dPdH[0:bgend])
    else:
        bgend = 0

    lowerbound = np.where(H<guessres-guesswidth*window)[0]
    if len(lowerbound) > 0:
        bgstart = lowerbound[0]
        bgH.extend(H[bgstart:])
        bg.extend(dPdH[bgstart:])
    else:
        bgstart = -1
    
    if len(bg) > 1:
        guess = np.array([0, 0])
        fit = curve_fit(BGfunc, bgH, bg, guess)
        fitBG = linearBG(H, *(fit[0]))
        plt.plot(H,dPdH,'.')
        plt.plot(H,fitBG)
        plt.vlines(H[bgend],np.min(dPdH),np.max(dPdH),linestyles='dotted')
        plt.vlines(H[bgstart],np.min(dPdH),np.max(dPdH),linestyles='dotted')
    else:
        fitBG = np.zeros_like(H)
    
    return fitBG

def fitFMR(H, dPdHoffset, guess, debug=False, guessplt=1, fixedphase=False, posdef=False):
    if fixedphase:
        fixargs = np.append(np.delete(np.arange(0,len(guess)),slice(2,None,4)),2)
        guessfix = guess[fixargs]
        upper = [np.inf]*len(guessfix)
        lower = [-np.inf]*len(guessfix)
        if posdef:
            lower[:-1:3] = [0.]*(len(guessfix)/3) #peak magnitude must be positive
        fit = curve_fit(lambda x,*guessfix: LorentzianDerivativeNWrapper(x, np.insert(guessfix[:-1],slice(2,None,3),guessfix[-1])), H, dPdHoffset, guessfix, bounds = (lower, upper))
        fitarg = np.insert(fit[0][:-1],slice(2,None,3),fit[0][-1])
    else:
        upper = [np.inf]*len(guess)
        lower = [-np.inf]*len(guess)
        if posdef:
            lower[::4] = [0.]*(len(guess)/4) #peak magnitude must be positive
            lower[2::4] = [0.]*(len(guess)/4) #peak phase must be between 0 and pi radians
            upper[2::4] = [np.pi]*(len(guess)/4) #TODO: fix peak output to say phase is in radians, not degrees
        fit = curve_fit(lambda x,*guess: LorentzianDerivativeNWrapper(x,guess), H, dPdHoffset, guess)
        fitarg = fit[0]
    fitY = LorentzianDerivativeNWrapper(H, fitarg)
    fitsep = np.array(fitarg)
    fitsep.shape = (fitsep.shape[0]/4,4)

    fitvar = np.array([c[i] for i,c in enumerate(fit[1])])
    if fixedphase:
        varsep = np.insert(fitvar[:-1],slice(2,None,3),fitvar[-1])
    else:
        varsep = fitvar
    varsep.shape = (varsep.shape[0]/4,4)

    #covsep = np.array(fit[1])
    #covsep.shape = (covsep.shape[0]/4,4)
    #TODO: fix for fixed specific variables

    fitsepY = [LorentzianDerivative(H, *(f)) for f in fitsep]
    
    plt.plot(H,dPdHoffset,'.')
    if debug:
    	plt.plot(H,LorentzianDerivativeNWrapper(H, guess)*guessplt)
    plt.plot(H,fitY,linewidth=3.0)
    residual = dPdHoffset-fitY
    plt.plot(H,residual+(np.min(fitY)-np.max(residual))*1.5,'.-k')
    for f in fitsepY:
        plt.plot(H,f)
    plt.ylabel('Derivative of Absorbed Power (a.u.)')
    plt.xlabel('External Field (Oe)')
        
    return fitsep,varsep#,covsep