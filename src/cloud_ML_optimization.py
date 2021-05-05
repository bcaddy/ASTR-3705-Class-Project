from scipy.optimize import minimize
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.special import hyp2f1
from scipy.integrate import quad
from scipy.special import gamma
plt.rcParams.update({'font.size': 18})
plt.rcParams.update({'font.family': 'serif'})
plt.rcParams.update({'mathtext.default':'regular'})
plt.rcParams.update({'mathtext.fontset':'stixsans'})
plt.rcParams.update({'axes.linewidth': 1.5})
plt.rcParams.update({'xtick.direction':'in'})
plt.rcParams.update({'xtick.major.size': 5})
plt.rcParams.update({'xtick.major.width': 1.25 })
plt.rcParams.update({'xtick.minor.size': 2.5})
plt.rcParams.update({'xtick.minor.width': 1.25 })
plt.rcParams.update({'ytick.direction':'in'})
plt.rcParams.update({'ytick.major.size': 5})
plt.rcParams.update({'ytick.major.width': 1.25 })
plt.rcParams.update({'ytick.minor.size': 2.5})
plt.rcParams.update({'ytick.minor.width': 1.25 })

os.chdir(os.path.dirname(__file__))

res512, res1024, res2048 = np.load("../data/masses.npy", allow_pickle=True)

colors = ["steelblue", "darkslategray", "lightsteelblue"]
fcolors = ["rosybrow", "darkred", "indianred"]


def cell_vol_kpc(res):
    return (10.0 / float(res)) ** 3
    
res = 2048

def model(mass, theta):
    """Broken power law model with 4 free parameters used to model the
    behavior of the cloud mass probability distribution.
    """
    mpeak, a1, a2, delta = theta
    return (mass / mpeak) ** (-a1) * (0.5 * (1 + (mass / mpeak) ** (1 / delta))) ** ((a1 - a2) * delta)
    

def bpl_analytic(theta, m):
    # analytic form of integral of model for normalization
    # doesn't work, use numerical integral for normalization
    mpeak, a1, a2, delta = theta
    coeff1 = 2 ** ((a2 - a1) * delta)
    coeff2 = (m / mpeak) ** (1 - a1)
    coeff3 = m + mpeak
    coeff4 = (((m + mpeak) / mpeak) ** (1 / delta)) ** ((a1 - a2) * delta)
    coeff = coeff1 * coeff2 * coeff3 * coeff4 * gamma(1 - a1)
    return coeff * hyp2f1(1, 2 - a2, 2 - a1, -(m / mpeak))
    

def lnlike(theta, mass, xmin):
    """Compute the likelihood of the data given the model.
    """
    mpeak, a1, a2, delta = theta
    # calculate broken power-law model for the current set of parameters
    mod = model(mass, theta)
    A = quad(model, xmin, 10 ** 4, args=theta)[0]
    # A = bpl_analytic(theta, mass)
    likenorm = mod / A
    # calculate the likelihood
    likelihood = np.sum(np.log(likenorm))
    if not np.isfinite(likelihood):
            return -np.inf
    return likelihood
    

def lnprior(theta):
    mpeak, a1, a2, delta = theta
    if not np.logical_and(a1 >= a1low, a1 <= a1up):
        return -np.inf
    if not np.logical_and(a2 >= a2low, a2 <= a2up):
        return -np.inf
    if not np.logical_and(mpeak >= mpeaklow, mpeak <= mpeakup):
        return -np.inf
    if not np.logical_and(delta >= deltalow, delta <= deltaup):
        return -np.inf
    # assume a flat prior
    return 0.0
    

def lnprob(theta, mass, xmin):
    mpeak, a1, a2, delta = theta
    lp = lnprior(theta)
    if not np.isfinite(lp):
            return -np.inf
    return lp + lnlike(theta, mass, xmin)
    
masses = res2048 * cell_vol_kpc(res)
xmin = 0.1
massind = np.where(masses > xmin)
masses1 = masses[massind]

def minus_lnprob(theta, mass=masses1, xmin=xmin):
    mpeak, a1, a2, delta = theta
    lp = lnprior(theta)
    if not np.isfinite(lp):
            return -np.inf
    return -(lp + lnlike(theta, mass, xmin))

nbins = 100

fig = plt.figure(figsize=(9, 7))
n2048, bins2048, patches2048 = plt.hist(np.log10(masses1), bins=nbins, color=colors[-1], label=f"$R={res}$")
plt.xlabel("$\log M_\odot$")
plt.ylabel("N Clouds")

center_mass = (10 ** bins2048[1:] + 10 ** bins2048[:-1]) / 2.

theta0=(40, -0.5, 1, 1)

a1low, a1up = -3, -0.01
a2low, a2up = 0.01, 3
mpeaklow, mpeakup = 5, 100
deltalow, deltaup = 0.1, 5

params = minimize(minus_lnprob, theta0, bounds=((mpeaklow, mpeakup),
                                                (a1low, a1up),
                                                (a2low, a2up),
                                                (deltalow, deltaup)),
                  args=(masses1, xmin))

print(params)

def ML_plotter(params, mass=center_mass, n=n2048):
    fig = plt.figure(figsize=(10, 8))
    data_norm = n2048 / np.sum(n2048)
    plt.loglog(mass, n2048, label="2048", color=colors[-1])
    # bin width in solar masses
    dM = (10 ** bins2048[1:] - 10 ** bins2048[:-1])
    theta = params.x
    pM = model(mass, theta)
    A = quad(model, xmin, 10 ** 4, args=theta)[0]
    expectation_value = pM * np.sum(n2048) * dM / A
    plt.loglog(mass, expectation_value, c=fcolors[-1], zorder=0)
    plt.plot(0, label="ML", color=fcolors[-1])
    plt.legend()
    plt.xlabel("Mass$~[M_\odot]$")
    plt.ylabel("N")
    plt.show()
    
ML_plotter(params)
