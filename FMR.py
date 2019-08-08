import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

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

def LorentzianDerivative(x, a, x0, theta, b):
    denom = (b**2 + (x-x0)**2)**2
    sym = -2*(x-x0)*b
    asym = (b**2-(x-x0)**2)
    return a/denom*(sym*np.cos(theta)+asym*np.sin(theta))

def LorentzianDerivativeN(x, a, x0, theta, b):
    return np.sum([LorentzianDerivative(x,a[i],x0[i],theta[i],b[i]) for i,x0i in enumerate(x0)],axis=0)

def LorentzianDerivativeNWrapper(x, *args):
    a = args[0][::4]
    x0 = args[0][1::4]
    theta = args[0][2::4]
    b = args[0][3::4]
    return LorentzianDerivativeN(x,a,x0,theta,b)

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

def fitFMR(H, dPdHoffset, guess, debug=False, guessplt=1):
    fit = curve_fit(lambda x,*guess: LorentzianDerivativeNWrapper(x,guess), H, dPdHoffset, guess)
    fitY = LorentzianDerivativeNWrapper(H, fit[0])
    fitsep = np.array(fit[0])
    fitsep.shape = (fitsep.shape[0]/4,4)
    varsep = np.array([c[i] for i,c in enumerate(fit[1])])
    varsep.shape = (varsep.shape[0]/4,4)
    #covsep = np.array(fit[1])
    #covsep.shape = (covsep.shape[0]/4,4)
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