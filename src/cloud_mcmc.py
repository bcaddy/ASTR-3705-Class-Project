import matplotlib.pyplot as plt
import numpy as np
import os
import emcee
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

def model(mass, theta):
    """Broken power law model with 4 free parameters used to model the
    behavior of the cloud mass probability distribution.
    """
    mpeak, a1, a2, delta = theta
    return (mass / mpeak) ** (-a1) * (0.5 * (1 + (mass / mpeak) ** (1 / delta))) ** ((a1 - a2) * delta)


def cloud_mcmc(masses, res, theta0=(40, -0.5, 1, 1), theta_bounds=((5, 100), (-2, -0.0001), (0.01, 3), (0.1, 5)),
               run_mcmc=False, xmin=0.1, color="lightsteelblue", fcolor="indianred"):
    resn = masses
    mpeak, a1, a2, delta = theta0
    mpeaklow, mpeakup = theta_bounds[0]
    a1low, a1up = theta_bounds[1]
    a2low, a2up = theta_bounds[2]
    deltalow, deltaup = theta_bounds[3]
    
    def cell_vol_kpc(res):
        return (10.0 / float(res)) ** 3

    masses = resn * cell_vol_kpc(res)
    massind = np.where(masses > xmin)
    masses1 = masses[massind]

    nbins = 100

    fig = plt.figure(figsize=(9, 7))
    n, bins, patches = plt.hist(np.log10(masses1), bins=nbins, color=color, label=f"$R={res}$")
    plt.xlabel("$\log M_\odot$")
    plt.ylabel("N Clouds")
    plt.close()
    
    center_mass = (10 ** bins[1:] + 10 ** bins[:-1]) / 2.
    
    fm = 3e4 * model(center_mass, theta0)

    fig = plt.figure(figsize=(9, 7))
    plt.plot(center_mass, n, label=f"$R={res}$", color=color)
    plt.plot(center_mass, fm, color=fcolor)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Mass$~M_\odot$")
    plt.ylabel("N")
    plt.legend()
    plt.close()
    
    # begin MCMC implementation
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
    
    data = (masses1, xmin)
    nwalkers = 50
    niter = 100
    theta0 = np.array([mpeak, a1, a2, delta])
    ndim = len(theta0)
    p0 = [np.array(theta0) * (1 + 0.1 * np.random.randn(ndim)) for i in range(nwalkers)]
               
    def main(p0, nwalkers, niter, ndim, lnprob, data):
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=data)

        print("Running burn-in...")
        p0, _, _ = sampler.run_mcmc(p0, 100)
        sampler.reset()

        print("Running production...")
        pos, prob, state = sampler.run_mcmc(p0, niter)

        return sampler, pos, prob, state
    
    
    if run_mcmc:
        sampler, pos, prob, state = main(p0, nwalkers, niter, ndim, lnprob, data)
   
    if run_mcmc:
        plotter(sampler, center_mass, n, color, fcolor, bins, xmin, res)
    
    if run_mcmc:
        return (sampler, bins, center_mass, masses1, n, fm)
    else:
        return (None, bins, center_mass, masses1, n, fm)


def plotter(sampler, mass, n, color, fcolor, bins, xmin, res):
    fig = plt.figure(figsize=(10, 8))
    data_norm = n / np.sum(n)
    plt.loglog(mass, n, color="k", linewidth=6, zorder=2)
    plt.loglog(mass, n, label=f"R={res} Data", color=color, linewidth=4, zorder=3)
    samples = sampler.flatchain
    # bin width in solar masses
    dM = (10 ** bins[1:] - 10 ** bins[:-1])
    mpeaks = []
    a1s = []
    a2s = []
    deltas = []
    theta_max = samples[np.argmax(sampler.flatlnprobability)]
    pMmax = model(mass, theta_max)
    Amax = quad(model, xmin, 10 ** 4, args=theta_max)[0]
    expectation_value_max = pMmax * np.sum(n) * dM / Amax
    plt.loglog(mass, expectation_value_max, label=f"R={res} Best-fit Model", color="lightcoral", linewidth=3, zorder=1)

    for theta in samples[np.random.randint(len(samples), size=1000)]:
        pM = model(mass, theta)
        A = quad(model, xmin, 10 ** 4, args=theta)[0]
        expectation_value = pM * np.sum(n) * dM / A
        plt.loglog(mass, expectation_value, c=fcolor, alpha=0.1, zorder=0)
        mpeaks.append(theta[0])
        a1s.append(theta[1])
        a2s.append(theta[2])
        deltas.append(theta[3])
    plt.plot(0, label="MCMC", color=fcolor)
    plt.legend()
    plt.xlabel("Mass$~[M_\odot]$")
    plt.ylabel("N")
    plt.savefig(f"../figures/mcmc_{res}.pdf")
    plt.show()

