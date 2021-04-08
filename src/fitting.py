import matplotlib.pyplot as plt
import numpy as np
import os
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

# this isn't quite right yet
# need initial guesses for different resolutions of cloud data

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

for i, r in enumerate(n):
    x = (bins[i][1:] + bins[i][:-1]) / 2.
    mass = 10 ** x

    # smoothly broken power law 1D fitting
    # https://docs.astropy.org/en/stable/api/astropy.modeling.powerlaws.SmoothlyBrokenPowerLaw1D.html

    # parameters
    a1 = -0.5
    a2 = 1
    mpeak = 40.
    delta = 1
    A = 3e4

    fm = A * (mass / mpeak) ** (-a1) * (0.5 * (1 + (mass / mpeak) ** (1 / delta))) ** ((a1 - a2) / delta)

    fig = plt.figure(figsize=(10, 8))
    plt.plot(10 ** x, n[i], label=f"$R={res[i]}$", color=colors[i])
    plt.plot(mass, fm, color=fcolors[i])
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.show()
