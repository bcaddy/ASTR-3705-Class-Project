import matplotlib.pyplot as plt
import numpy as np
import os
import emcee
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

# resource: https://prappleizer.github.io/Tutorials/MCMC/MCMC_Tutorial.html
# Jeff's notebook is also helpful

os.chdir(os.path.dirname(__file__))

res512, res1024, res2048 = np.load("../data/masses.npy", allow_pickle=True)

res = [512, 1024, 2048]
masses = [res512, res1024, res2048]
colors = ["steelblue", "darkslategray", "lightsteelblue"]
fcolors =  ["firebrick", "lightcoral", "indianred"]

nbins = 100

def cell_vol_kpc(res):
    return (10.0 / float(res)) ** 3

n2048, bins2048, patches2048 = plt.hist(np.log10(res2048 * cell_vol_kpc(res[2])), bins=nbins, color=colors[2], label=r"$R={2048}$")
n1024, bins1024, patches1024 = plt.hist(np.log10(res1024 * cell_vol_kpc(res[1])), bins=nbins, color=colors[1], label=r"$R={1024}$")
n512, bins512, patches512 = plt.hist(np.log10(res512 * cell_vol_kpc(res[0])), bins=nbins, color=colors[0], label=r"$R={512}$")
plt.close()


n = [n512, n1024, n2048]
bins = [bins512, bins1024, bins2048]
patches = [patches512, patches1024, patches2048]

# parameters
# index 2 has parameters from Jeff
# other indices are values that I just made up and are def wrong
# need to figure out how to come up with initial guess for paramaters
a1n = [1e-10, 1e-10, -0.5]
a2n = [1e-10, 1e-10, 1]
mpeakn = [1e-10, 1e-10, 40.]
deltan = [1, 1, 1]
An = [1e-10, 1e-10, 3e4]

for i, r in enumerate(n):
    x = (bins[i][1:] + bins[i][:-1]) / 2.
    mass = 10 ** x

# try to implement MCMC for R = 2048 cloud data
a1 = a1n[2]
a2 = a2n[2]
mpeak = mpeakn[2]
delta = deltan[2]
A = An[2]

fm = A * (mass / mpeak) ** (-a1) * (0.5 * (1 + (mass / mpeak) ** (1 / delta))) ** ((a1 - a2) / delta)

fig = plt.figure(figsize=(10, 8))
plt.plot(10 ** x, n[2], label=f"$R={res[2]}$", color=colors[2])
plt.plot(mass, fm, color=fcolors[2])
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.show()

def model(mass, theta):
    """Broken power law model with 4 free parameters used to model the
    behavior of the cloud mass probability distribution.
    """
    mpeak, a1, a2, delta = theta
    return (mass / mpeak) ** (-a1) * (0.5 * (1 + (mass / mpeak) ** (1 / delta))) ** ((a1 - a2) * delta)

def mod_sq(mass, theta):
    # function to integrate to normalize PDF
    return ((mass / mpeak) ** (-a1) * (0.5 * (1 + (mass / mpeak) ** (1 / delta))) ** ((a1 - a2) * delta)) ** 2

def lnlike(theta, mass, dist, xmin):
    """Compute the likelihood of the data given the model.
    
    mod is the model.
    mass is the mass bins.
    dist is the number of clouds per mass bin.
    xmin is the minimum value of mass for which the power law behavior holds.
    
    ATM: normalization is bunk and not sure if likelihood is correct
    """
    mpeak, a1, a2, delta = theta
    # calculate broken power-law model for the current set of parameters
    mod = model(mass, theta)
    A_sq = quad(model, xmin, np.inf, args=(theta))[0]
    likenorm = mod / np.sqrt(A_sq)
    # calculate the likelihood
    likelihood = np.sum(np.log(likenorm))
    if not np.isfinite(likelihood):
            return -np.inf
    return likelihood

a1low, a1up = -1000, -1e-9
a2low, a2up = 1e-9, 1000
mpeaklow, mpeakup = 1e-9, 1000
deltalow, deltaup = 1e-9, 1000
def lnprior(theta):
    mpeak, a1, a2, delta = theta
    if not (a1 >= a1low, a1 <= a1up):
        return -np.inf
    if not (a2 >= a2low, a2 <= a2up):
        return -np.inf
    if not (mpeak >= mpeaklow, mpeak <= mpeakup):
        return -np.inf
    if not (delta >= deltalow, delta <= deltaup):
        return -np.inf
    # assume a flat prior
    return 0.0

def lnprob(theta, mass, dist, xmin):
    mpeak, a1, a2, delta = theta
    lp = lnprior(theta)
    if not np.isfinite(lp):
            return -np.inf
    return lp + lnlike(theta, mass, dist, xmin)


x2048 = (bins2048[1:] + bins2048[:-1]) / 2.
mass2048 = 10 ** x2048
n_norm = [float(i) / sum(n2048[36:]) for i in n2048[36:]]

# mass bins, number of clouds, xmin
data = (mass2048[36:], n_norm, 1)
nwalkers = 1000
niter = 100
# mpeak, a1, a2, delta
theta0 = np.array([20, -0.5, 0.5, 1])
ndim = len(theta0)
p0 = [np.array(theta0) + 1e-7 * np.random.randn(ndim) for i in range(nwalkers)]

def main(p0, nwalkers, niter, ndim, lnprob, data):
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=data)

    print("Running burn-in...")
    p0, _, _ = sampler.run_mcmc(p0, 100)
    sampler.reset()

    print("Running production...")
    pos, prob, state = sampler.run_mcmc(p0, niter)

    return sampler, pos, prob, state

sampler, pos, prob, state = main(p0, nwalkers, niter, ndim, lnprob, data)

def plotter(sampler, mass=mass2048[36:], n=n_norm):
    plt.loglog(mass, n)
    samples = sampler.flatchain
    for theta in samples[np.random.randint(len(samples), size=100)]:
        plt.loglog(mass, model(mass, theta), color="r", alpha=0.1)
    plt.xlabel(r"Mass$~[M_\odot]$")
    plt.ylabel("N Clouds")
    plt.show()
    
plotter(sampler)
