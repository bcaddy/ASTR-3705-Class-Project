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

# this isn't quite working yet
# need to figure out what condiditons to put on lnprior
def model(theta, mass=mass):
    A, mpeak, a1, a2, delta = theta
    return A * (mass / mpeak) ** (-a1) * (0.5 * (1 + (mass / mpeak) ** (1 / delta))) ** ((a1 - a2) / delta)

def lnlike(theta, x, y, yerr):
    A, mpeak, a1, a2, delta = theta
    return - (1 / 2) * np.sum(y - model(theta) / yerr) ** 2

def lnprior(theta):
    A, mpeak, a1, a2, delta = theta
    if conditions:
        return 0.0
    else:
        return -np.inf

def lnprob(theta, x, y, yerr):
    A, mpeak, a1, a2, delta = theta
    lp = lnprior(theta, x, y, yerr)
    if not np.isfinite(lp):
        return - np.inf
    return lp + lnlike(theta, x, y, yerr)


nerr = 0.05 * n[2]
data = (mass, n[2], nerr)
nwalkers = 1000
niter = 100
initial = np.array([An[2], mpeakn[2], a1n[2], a2n[2], deltan[2]])
ndim = len(initial)
p0 = [np.array(initial) + 1e-7 * np.random.randn(ndim) for i in range(nwalkers)]

# function that uses emcee to actually implement MCMC on the data
def main(p0, nwalkers, niter, ndim, lnprob, data):
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=data)

    print("Running burn-in...")
    p0, _, _ = sampler.run_mcmc(p0, 100)
    sampler.reset()

    print("Running production...")
    pos, prob, state = sampler.run_mcmc(p0, niter)

    return sampler, pos, prob, state

sampler, pos, prob, state = main(p0, nwalkers, niter, ndim, lnprob, data)

def plotter(sampler, mass=mass, n=n[2]):
    plt.ion()
    plt.plot(age,T,label='Change in T')
    samples = sampler.flatchain
    for theta in samples[np.random.randint(len(samples), size=100)]:
        plt.plot(age, model(theta, age), color="r", alpha=0.1)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.xlabel('Years ago')
    plt.ylabel(r'$\Delta$ T (degrees)')
    plt.legend()
    plt.show()
