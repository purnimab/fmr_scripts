# -*- coding: utf-8 -*-
#from __future__ import division
import numpy as np
from scipy.optimize import curve_fit

h = 6.62607004e-34*1e7 #erg-s
muB = 9.274000999e-21 #erg/G
mu0 = 1. #G/Oe - field actually measured in G not Oe
giga = 1e9
muBmu0h = muB*mu0/h/giga
OetoAperm = 1000/(4*np.pi)

def OOPresfield(HFMR, g, Meff, **kwargs):
    if len(kwargs) > 0:
        print 'Unneccesary variables:'
        print kwargs

    gmuBmu0h = g*muBmu0h
    off = -4*np.pi*Meff*gmuBmu0h
    return OOPresfield2(HFMR, [gmuBmu0h, off])

def OOPresfield2(HFMR,params):
    return np.poly1d(params)(HFMR)

def OOPlinewidth(f, deltaH0, alpha, g, **kwargs):
    if len(kwargs) > 0:
        print 'Unneccesary variables:'
        print kwargs

    gmuBmu0h = g*muBmu0h
    return OOPlinewidth2(f,[alpha/gmuBmu0h, deltaH0])

def OOPlinewidth2(f,params):
    return np.poly1d(params)(f)

def IPresfield(HFMR, g, Hcubic, angle, Meff, **kwargs):
    if len(kwargs) > 0:
        print 'Unneccesary variables:'
        print kwargs

    gmuBmu0h = g*muBmu0h
    return IPresfield2(HFMR,gmuBmu0h,Hcubic,angle,Meff)

def IPresfield2(HFMR,factor,Hcubic,angle,Meff):
    return factor*np.sqrt((HFMR+Hcubic*np.cos(4.*angle))*(HFMR+4.*np.pi*Meff+Hcubic/4.*(3.+np.cos(4.*angle))))

def IPresfield2dir(HFMR, n1, n2, g, Hcubic, angle1, angle2, Meff, **kwargs):
    if len(HFMR) != n1 + n2:
        raise IndexError('Length of dataset does not match input split')

    if len(kwargs) > 0:
        print 'Unneccesary variables:'
        print kwargs

    x1 = HFMR[:n1]
    x2 = HFMR[-n2:]
    y1 = IPresfield(x1,g,Hcubic,angle1,Meff)
    y2 = IPresfield(x2,g,Hcubic,angle2,Meff)
    return np.concatenate([y1,y2])

def vsfreq(x, n, orientation, g, Hcubic, Meff, deltaH0, alpha, Ms, *args):
    #x is a concatenated list of HFMR, freq
    #n are the lengths of HFMR (same as the next freq list)
    #orientation are booleans, True if IP, False if OOP
    #*args are angles of length of # of IP data

    if len(x) != np.sum(n)*2:
        raise IndexError('Length of dataset does not match input split')
    ip = np.sum(orientation)
    if ip == 1:
        angles = args[:-1]
        if ip != len(angles):
            raise IndexError('Number of IP datasets does not match fit parameters')
        G = args[-1]
    else:
        angles = args[:-2]
        if ip != len(angles):
            raise IndexError('Number of IP datasets does not match fit parameters')
        G100 = args[-2]
        G110 = args[-1]

    j = 0
    y = []
    m = np.insert(np.cumsum(np.array(n)*2),0,0)
    for i in np.arange(0,len(n)):
        h = x[m[i]:m[i]+n[i]]
        f = x[m[i+1]-n[i]:m[i+1]]
        #x = f[m[i]:m[i+1]]
        if orientation[i]:
            angle = angles[j]
            j += 1
            y = np.append(y,IPresfield(h,g,Hcubic,angle,Meff))
            if ip == 1:
                y = np.append(y,IPlinewidth1dir(f,deltaH0,alpha,g,Ms,G))
            else:
                y = np.append(y,IPlinewidth(f,deltaH0,alpha,g,angle,Ms,G100,G110))
        else:
            y = np.append(y,OOPresfield(h,g,Meff))
            y = np.append(y,OOPlinewidth(f,deltaH0,alpha,g))

    return y

def resfield(HFMR, n, orientation, g, Hcubic, Meff, *args):
    #HFMR is a concatenated list
    #n are the sequences
    #orientation are booleans, True if IP, False if OOP
    #*args are angles of length of # of IP data
    if len(HFMR) != np.sum(n):
        raise IndexError('Length of dataset does not match input split')
    if np.sum(orientation) != len(args):
        raise IndexError('Number of IP datasets does not match fit parameters')

    j = 0
    y = []
    m = np.insert(np.cumsum(n),0,0)
    for i in np.arange(0,len(n)):
        x = HFMR[m[i]:m[i+1]]
        if orientation[i]:
            #In-plane
            angle = args[j]
            j += 1
            y = np.append(y,IPresfield(x,g,Hcubic,angle,Meff))
        else:
            y = np.append(y,OOPresfield(x,g,Meff))
    return y

def IPlinewidth(f, deltaH0, alpha, g, angle, Ms, G100, G110, **kwargs):
    if len(kwargs) > 0:
        print 'Unneccesary variables:'
        print kwargs

    G = G100*np.cos(2.*angle)**2 + G110*np.cos(2.*(angle-np.pi/4.))**2
    return IPlinewidth1dir(f, deltaH0, alpha, g, Ms, G)

def IPlinewidth1dir(f, deltaH0, alpha, g, Ms, G, **kwargs):
    if len(kwargs) > 0:
        print 'Unneccesary variables:'
        print kwargs

    gmuBmu0h = g*muBmu0h
    fm = gmuBmu0h*4.*np.pi*Ms
    sqrt = np.sqrt(f**2+(fm/2.)**2)
    mag = np.arcsin(np.sqrt((sqrt-fm/2.)/(sqrt+fm/2.)))
    return deltaH0 + alpha/gmuBmu0h*f + G*mag

def IPlinewidth2dir(f, n1, n2, deltaH0, alpha, g, angle1, angle2, Ms, G100, G110, **kwargs):
    if len(f) != n1 + n2:
        raise IndexError('Length of dataset does not match input split')

    x1 = f[:n1]
    x2 = f[-n2:]
    y1 = IPlinewidth(x1,deltaH0,alpha,g,angle1,Ms,G100,G110)
    y2 = IPlinewidth(x2,deltaH0,alpha,g,angle2,Ms,G100,G110)
    return np.concatenate([y1,y2])

def linewidth(f, n, orientation, deltaH0, alpha, g, Ms, *args):
    #f is a concatenated list
    #n are the sequences
    #orientation are booleans, True if IP, False if OOP
    #*args are angles of length of # of IP data concatenated with either G or [G100,G110] depending number of IP scans
    if len(f) != np.sum(n):
        raise IndexError('Length of dataset does not match input split')
    ip = np.sum(orientation)
    if ip == 1:
        angles = args[:-1]
        if ip != len(angles):
            raise IndexError('Number of IP datasets does not match fit parameters')
        G = args[-1]
    elif ip != 0:
        angles = args[:-2]
        if ip != len(angles):
            raise IndexError('Number of IP datasets does not match fit parameters')
        G100 = args[-2]
        G110 = args[-1]

    j = 0
    y = []
    m = np.insert(np.cumsum(n),0,0)
    for i in np.arange(0,len(n)):
        x = f[m[i]:m[i+1]]
        if orientation[i]:
            angle = angles[j]
            j += 1
            if ip == 1:
                y = np.append(y,IPlinewidth1dir(x,deltaH0,alpha,g,Ms,G))
            else:
                y = np.append(y,IPlinewidth(x,deltaH0,alpha,g,angle,Ms,G100,G110))
        else:
            y = np.append(y,OOPlinewidth(x,deltaH0,alpha,g))

    return y


#def IPlinewidth2(f,H0,alpha,ms,g100,g110):
#    angle = HFMRfit[0][2]
#    gmuBmu0h = HFMRfit[0][0]
#    fm = gmuBmu0h*4.*np.pi*ms
#    sqrt = np.sqrt((f*giga)**2+(fm/2.)**2)
#    mag = np.arcsin(np.sqrt((sqrt-fm/2.)/(sqrt+fm/2.)))
#    return H0 + alpha/gmuBmu0h*f*giga + g100*mag*np.cos(2.*angle)**2 + g110*mag*np.cos(2.*(angle-np.pi/4.))**2

def fitOOP(f, HFMR, deltaH):
    #1. fit resonance field
    HFMRguess = [2.,320.]
    HFMRfit = curve_fit(OOPresfield,HFMR,f,HFMRguess)
    print HFMRfit
    g = HFMRfit[0][0]
    Meff = HFMRfit[0][1]
    #2. fit linewidth with fixed g calculated above
    deltaHguess = [10.,.001] #fitting out deltaH0, alpha
    deltaHfit = curve_fit(lambda f,*guess: OOPlinewidth(f,*np.append(guess,g)), f, deltaH, deltaHguess)
    print deltaHfit
    deltaH0 = deltaHfit[0][0]
    alpha = deltaHfit[0][1]
    return {'g':g, 'Meff':Meff, 'deltaH0':deltaH0, 'alpha':alpha}
    #return (g,Meff,deltaH0,alpha)

def fitIP(f, HFMR, deltaH):
    #1. fit resonance field
    HFMRguess = [2., -40., 0., 320.]
    HFMRfit = curve_fit(IPresfield,HFMR,f,HFMRguess)
    print HFMRfit
    g = HFMRfit[0][0]
    Hcub = HFMRfit[0][1]
    angle = HFMRfit[0][2] #radians
    Meff = HFMRfit[0][3]
    #2. fit linewidth, with g, angle fixed from above
    #zero = np.min(np.abs(angle-np.array([0.,np.pi/4.])))
    deltaHguess = [10.,.001,320.,200.]
    deltaHfit = curve_fit(lambda f,*guess: IPlinewidth1dir(f,*np.insert(guess,2,g)), f, deltaH, deltaHguess)
    print deltaHfit
    deltaH0 = deltaHfit[0][0]
    alpha = deltaHfit[0][1]
    Ms = deltaHfit[0][2]
    G = deltaHfit[0][3]
    return {'g':g, 'Hcubic':Hcub, 'angle':np.remainder(angle,2*np.pi), 'Meff':Meff, 'deltaH0':deltaH0, 'alpha':alpha, 'Ms':Ms, 'G':G} #we only know G110 or G100 if angle = 0 or np.pi/4.
    #return (g, Hcub, angle, Meff, deltaH0, alpha, Ms, G) #we only know G110 or G100 if angle = 0 or np.pi/4.

def calcAngs(angTypes,anglist,params):
    angs = np.zeros(len(angTypes))
    j = 0
    for i,t in enumerate(angTypes):
        if t == 1: #approximate
            angs[i] = params[j]
            j += 1
        else:
            angs[i] = anglist[i] + float(params[-1])
    return angs

def fitAll(f, HFMR, deltaH, n, direction, *args):
    #f,HFMR,deltaH are concatenated lists
    #n are the sequences
    #direction is list of:
    #   2 if IP mounted on rotator (angle relative to one another, can vary slightly
    #   1 if IP measurement (fixed angles since they can't be determined properly) - overfitting
    #   0 if OOP measurement
    #args are the guesstimates of the angle (from rotator, or 110 = 45deg)
    if len(set([len(f), len(HFMR), len(deltaH), np.sum(n)])) != 1: #check if all are equal
        raise IndexError('Length of datasets, input split, incompatible')
    if len(n) != len(direction):
        raise IndexError('Input split incompatible')
    orientation = [i != 0 for i in direction] #True if IP, False if OOP
    ip = np.sum(orientation)
    if ip > 0:
        angs = args[0]
    else:
        angs = []

    if ip != len(angs):
        raise IndexError('IP parameters incompatible')
    if ip > 0:
        angTypes = np.array(direction)[np.where(orientation)[0]]
        approx = [i == 1 for i in angTypes]
        relative = [i == 2 for i in angTypes]

    results = {}

    #1. fit resonance field
    if ip == 0:
        HFMRguess = [2.,320.]
        Hcub = np.nan
        HFMRfit = curve_fit(lambda HFMR,*guess: resfield(HFMR,n,orientation,*np.insert(guess,1,Hcub)), HFMR, f, HFMRguess)
        g = HFMRfit[0][0]
        Meff = HFMRfit[0][1]
        angles = np.nan
    elif ip < 3: #if there are only 2 angles, that is hard to fit so they must be fixed
        HFMRguess = [2.,-40.,320.]
        angles = angs
        HFMRfit = curve_fit(lambda HFMR,*guess: resfield(HFMR,n,orientation,*np.concatenate([guess,angles])), HFMR, f, HFMRguess)
        g = HFMRfit[0][0]
        Hcub = HFMRfit[0][1]
        Meff = HFMRfit[0][2]
        results.update({'Hcubic':Hcub})
    else:
        HFMRguessNoAng = [2.,-40.,320.]
        angGuess = []
        if np.sum(approx) > 0:
            angGuess = np.concatenate([angGuess,np.array(angs)[np.where(approx)[0]]])
        if np.sum(relative) > 0:
            angGuess = np.append(angGuess,0.)
        print 'free angles:'
        print len(angGuess)
        print angGuess
        print angGuess+np.pi/8
        HFMRguess = np.concatenate([HFMRguessNoAng,angGuess])
        #bounds = (np.concatenate(([-np.inf]*len(HFMRguessNoAng),np.array(angGuess)-np.deg2rad(5.))),np.concatenate(([np.inf]*len(HFMRguessNoAng),np.array(angGuess)+np.deg2rad(5.))))
        HFMRfit = curve_fit(lambda HFMR,*guess: resfield(HFMR,n,orientation,*np.concatenate([guess[:3],calcAngs(angTypes,angs,guess[3:])])), HFMR, f, HFMRguess)#, bounds=bounds)
        g = HFMRfit[0][0]
        Hcub = HFMRfit[0][1]
        Meff = HFMRfit[0][2]

        angles = calcAngs(angTypes,angs,HFMRfit[0][3:])

        results.update({'Hcubic':Hcub, 'angles':angles})
    results.update({'g':g, 'Meff':Meff})

    #2. fit linewidth with fixed g, angles from above
    if ip == 0:
        deltaHguess = [10.,.001]
        deltaHfit = curve_fit(lambda f,*guess: linewidth(f,n,orientation,*np.concatenate([guess,[g,Meff]])), f, deltaH, deltaHguess)
        deltaH0 = deltaHfit[0][0]
        alpha = deltaHfit[0][1]
    elif ip == 1:
        deltaHguess = [10.,.001,320.,200.]
        deltaHfit = curve_fit(lambda f,*guess: linewidth(f,n,orientation,*np.insert(np.insert(guess,2,g),-1,angles)), f, deltaH, deltaHguess)
        deltaH0 = deltaHfit[0][0]
        alpha = deltaHfit[0][1]
        Ms = deltaHfit[0][2]
        G = deltaHfit[0][3]
        results.update({'Ms':Ms, 'G':G})
    else:
        deltaHguess = [10.,.001,320.,200.,400.]
        deltaHfit = curve_fit(lambda f,*guess: linewidth(f,n,orientation,*np.insert(np.insert(guess,2,g),-2,angles)), f, deltaH, deltaHguess)
        deltaH0 = deltaHfit[0][0]
        alpha = deltaHfit[0][1]
        Ms = deltaHfit[0][2]
        G100 = deltaHfit[0][3]
        G110 = deltaHfit[0][4]
        results.update({'Ms':Ms, 'G100':G100, 'G110':G110})
    results.update({'deltaH0':deltaH0, 'alpha':alpha})
    print HFMRfit
    print deltaHfit

    return results #only contains fitted parameters
    #return (g, Hcub, angles, Meff, deltaH0, alpha, Ms, G)

def fitAllWithErrors(f, HFMR, deltaH, errHFMR, errdeltaH, n, direction, *args):
    #f,HFMR,deltaH are concatenated lists
    #n are the sequences
    #direction is list of:
    #   2 if IP mounted on rotator (angle relative to one another, can vary slightly
    #   1 if IP measurement (fixed angles since they can't be determined properly) - overfitting
    #   0 if OOP measurement
    #args are the guesstimates of the angle (from rotator, or 110 = 45deg)
    if len(set([len(f), len(HFMR), len(deltaH), len(errHFMR), len(errdeltaH), np.sum(n)])) != 1: #check if all are equal
        raise IndexError('Length of datasets, input split, incompatible')
    if len(n) != len(direction):
        raise IndexError('Input split incompatible')
    orientation = [i != 0 for i in direction] #True if IP, False if OOP
    ip = np.sum(orientation)
    if ip > 0:
        angs = args[0]
    else:
        angs = []

    if ip != len(angs):
        raise IndexError('IP parameters incompatible')
    if ip > 0:
        angTypes = np.array(direction)[np.where(orientation)[0]]
        approx = [i == 1 for i in angTypes]
        relative = [i == 2 for i in angTypes]

    results = {}
    errors = {}

    #1. fit resonance field TODO: figure out how to incorporate errors in x (HFMR) into this fitting - but these are much lower so maybe not an issue
    if ip == 0:
        HFMRguess = [2.,320.]
        Hcub = np.nan
        HFMRfit = curve_fit(lambda HFMR,*guess: resfield(HFMR,n,orientation,*np.insert(guess,1,Hcub)), HFMR, f, HFMRguess, absolute_sigma=True)
        g = HFMRfit[0][0]
        gerr = HFMRfit[1][0][0]
        Meff = HFMRfit[0][1]
        Mefferr = HFMRfit[1][1][1]
        angles = np.nan
    elif ip < 3: #if there are only 2 angles, that is hard to fit so they must be fixed
        HFMRguess = [2.,-40.,320.]
        angles = angs
        HFMRfit = curve_fit(lambda HFMR,*guess: resfield(HFMR,n,orientation,*np.concatenate([guess,angles])), HFMR, f, HFMRguess, absolute_sigma=True)
        g = HFMRfit[0][0]
        gerr = HFMRfit[1][0][0]
        Hcub = HFMRfit[0][1]
        Hcuberr = HFMRfit[1][1][1]
        Meff = HFMRfit[0][2]
        Mefferr = HFMRfit[1][2][2]
        results.update({'Hcubic':Hcub})
        errors.update({'errHcubic':Hcuberr})
    else:
        HFMRguessNoAng = [2.,-40.,320.]
        angGuess = []
        if np.sum(approx) > 0:
            angGuess = np.concatenate([angGuess,np.array(angs)[np.where(approx)[0]]])
        if np.sum(relative) > 0:
            angGuess = np.append(angGuess,0.)
        print 'free angles:'
        print len(angGuess)
        print angGuess
        print angGuess+np.pi/8
        HFMRguess = np.concatenate([HFMRguessNoAng,angGuess])
        #bounds = (np.concatenate(([-np.inf]*len(HFMRguessNoAng),np.array(angGuess)-np.deg2rad(5.))),np.concatenate(([np.inf]*len(HFMRguessNoAng),np.array(angGuess)+np.deg2rad(5.))))
        HFMRfit = curve_fit(lambda HFMR,*guess: resfield(HFMR,n,orientation,*np.concatenate([guess[:3],calcAngs(angTypes,angs,guess[3:])])), HFMR, f, HFMRguess, absolute_sigma=True)#, bounds=bounds)
        g = HFMRfit[0][0]
        gerr = HFMRfit[1][0][0]
        Hcub = HFMRfit[0][1]
        Hcuberr = HFMRfit[1][1][1]
        Meff = HFMRfit[0][2]
        Mefferr = HFMRfit[1][2][2]

        angles = calcAngs(angTypes,angs,HFMRfit[0][3:])
        angleserr = calcAngs(angTypes,np.array([0]*len(angTypes)),np.diag(HFMRfit[1])[3:])

        results.update({'Hcubic':Hcub, 'angles':angles})
        errors.update({'errHcubic':Hcuberr, 'errangles':angleserr})
    results.update({'g':g, 'Meff':Meff})
    errors.update({'errg':gerr, 'errMeff':Mefferr})

    #2. fit linewidth with fixed g, angles from above
    if ip == 0:
        deltaHguess = [10.,.001]
        deltaHfit = curve_fit(lambda f,*guess: linewidth(f,n,orientation,*np.concatenate([guess,[g,Meff]])), f, deltaH, deltaHguess, sigma=errdeltaH, absolute_sigma=True)
        deltaH0 = deltaHfit[0][0]
        deltaH0err = deltaHfit[1][0][0]
        alpha = deltaHfit[0][1]
        alphaerr = deltaHfit[1][1][1]
    elif ip == 1:
        deltaHguess = [10.,.001,320.,200.]
        deltaHfit = curve_fit(lambda f,*guess: linewidth(f,n,orientation,*np.insert(np.insert(guess,2,g),-1,angles)), f, deltaH, deltaHguess, sigma=errdeltaH, absolute_sigma=True)
        deltaH0 = deltaHfit[0][0]
        deltaH0err = deltaHfit[1][0][0]
        alpha = deltaHfit[0][1]
        alphaerr = deltaHfit[1][1][1]
        Ms = deltaHfit[0][2]
        Mserr = deltaHfit[1][2][2]
        G = deltaHfit[0][3]
        Gerr = deltaHfit[1][3][3]
        results.update({'Ms':Ms, 'G':G})
        errors.update({'errMs':Mserr, 'errG':Gerr})
    else:
        deltaHguess = [10.,.001,320.,200.,400.]
        deltaHfit = curve_fit(lambda f,*guess: linewidth(f,n,orientation,*np.insert(np.insert(guess,2,g),-2,angles)), f, deltaH, deltaHguess, sigma=errdeltaH, absolute_sigma=True)
        deltaH0 = deltaHfit[0][0]
        deltaH0err = deltaHfit[1][0][0]
        alpha = deltaHfit[0][1]
        alphaerr = deltaHfit[1][1][1]
        Ms = deltaHfit[0][2]
        Mserr = deltaHfit[1][2][2]
        G100 = deltaHfit[0][3]
        G100err = deltaHfit[1][3][3]
        G110 = deltaHfit[0][4]
        G110err = deltaHfit[1][4][4]
        results.update({'Ms':Ms, 'G100':G100, 'G110':G110})
        errors.update({'errMs':Mserr, 'errG100':G100err, 'errG110':G110err})
    results.update({'deltaH0':deltaH0, 'alpha':alpha})
    errors.update({'errdeltaH0':deltaH0err, 'erralpha':alphaerr})
    print HFMRfit
    print deltaHfit

    return results,errors #only contains fitted parameters
    #return (g, Hcub, angles, Meff, deltaH0, alpha, Ms, G)

def fitSimul(x, y, n, direction, *args): #TODO: actually finish this
    #x, y are concatenated lists of {HFMR->f} and {f->linewidth} respectively
    #n are the sequences (length of each list, but only one is listed)
    #direction is list of:
    #   2 if IP mounted on rotator (angle relative to one another, can vary slightly
    #   1 if IP measurement (angles not exactly relative)
    #   0 if OOP measurement
    #args are the guesstimates of the angle (from rotator, or 110 = 45deg)
    if len(set([len(x), len(y), np.sum(n)*2])) != 1: #check if all are equal
        raise IndexError('Length of datasets, input split, incompatible')
    if len(n) != len(direction):
        raise IndexError('Input split incompatible')
    orientation = [i != 0 for i in direction] #True if IP, False if OOP
    ip = np.sum(orientation)
    if ip > 0:
        angs = args[0]
    if ip != len(angs):
        raise IndexError('IP parameters incompatible')
    if ip > 0:
        angTypes = np.array(direction)[np.where(orientation)[0]]
        approx = [i == 1 for i in angTypes]
        relative = [i == 2 for i in angTypes]

    results = {}

    #generate guesses and perform

    #1. fit resonance field
    if ip == 0:
        #x, n, orientation, g, Hcubic, Meff, deltaH0, alpha, Ms, *args
        HFMRguess = [2.,320.,10,.001]
        Hcub = np.nan
        Ms = np.nan
        HFMRfit = curve_fit(lambda x,*guess: vsfreq(x,n,orientation,*np.append(np.insert(guess,1,Hcub),Ms)), x, y, guess)
        g = HFMRfit[0][0]
        Meff = HFMRfit[0][1]
        deltaH0 = HFMRfit[0][2]
        alpha = HFMRfit[0][3]
    #elif ip < 3: #if there are only 2 angles, that is hard to fit so they must be fixed
    #    HFMRguess = [2.,-40.,320.]
    #    angles = angs
    #    HFMRfit = curve_fit(lambda HFMR,*guess: resfield(HFMR,n,orientation,*np.concatenate([guess,angles])), HFMR, f, HFMRguess)
    #    g = HFMRfit[0][0]
    #    Hcub = HFMRfit[0][1]
    #    Meff = HFMRfit[0][2]
    #    results.update({'Hcubic':Hcub})
    elif ip == 1:
        guessNoAng = [2.,-40.,320.,10.,.001,320.,200.]

        #revised up to this point!
        angGuess = []
        if np.sum(approx) > 0:
            angGuess = np.concatenate([angGuess,np.array(angs)[np.where(approx)[0]]])
        if np.sum(relative) > 0:
            angGuess = np.append(angGuess,0.)
        print 'free angles:'
        print len(angGuess)
        print angGuess
        print angGuess+np.pi/8
        HFMRguess = np.concatenate([HFMRguessNoAng,angGuess])
        #bounds = (np.concatenate(([-np.inf]*len(HFMRguessNoAng),np.array(angGuess)-np.deg2rad(5.))),np.concatenate(([np.inf]*len(HFMRguessNoAng),np.array(angGuess)+np.deg2rad(5.))))
        HFMRfit = curve_fit(lambda HFMR,*guess: resfield(HFMR,n,orientation,*np.concatenate([guess[:3],calcAngs(angTypes,angs,guess[3:])])), HFMR, f, HFMRguess)#, bounds=bounds)
        g = HFMRfit[0][0]
        Hcub = HFMRfit[0][1]
        Meff = HFMRfit[0][2]

        angles = calcAngs(angTypes,angs,HFMRfit[0][3:])
        results.update({'Hcubic':Hcub, 'angles':angles})
    results.update({'g':g, 'Meff':Meff})

    #2. fit linewidth with fixed g, angles from above
    if ip == 1:
        deltaHguess = [10.,.001,320.,200.]
        deltaHfit = curve_fit(lambda f,*guess: linewidth(f,n,orientation,*np.insert(np.insert(guess,2,g),-1,angles)), f, deltaH, deltaHguess)
        deltaH0 = deltaHfit[0][0]
        alpha = deltaHfit[0][1]
        Ms = deltaHfit[0][2]
        G = deltaHfit[0][3]
        results.update({'Ms':Ms, 'G':G})
    else:
        deltaHguess = [10.,.001,320.,200.,400.]
        deltaHfit = curve_fit(lambda f,*guess: linewidth(f,n,orientation,*np.insert(np.insert(guess,2,g),-2,angles)), f, deltaH, deltaHguess)
        deltaH0 = deltaHfit[0][0]
        alpha = deltaHfit[0][1]
        Ms = deltaHfit[0][2]
        G100 = deltaHfit[0][3]
        G110 = deltaHfit[0][4]
        results.update({'Ms':Ms, 'G100':G100, 'G110':G110})
    results.update({'deltaH0':deltaH0, 'alpha':alpha})
    print HFMRfit
    print deltaHfit

    return results #only contains fitted parameters
