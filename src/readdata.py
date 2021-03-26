import matplotlib.pyplot as plt
import numpy as np
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

res512, res1024, res2048 = np.load("../data/jeff.npy", allow_pickle=True)

def cell_vol_kpc(res):
    return (10.0 / float(res)) ** 3

n, bins, patches = plt.hist(np.log10(res2048 * cell_vol_kpc(2048)), bins=100,
                            color="steelblue")
plt.yscale("log")
plt.xlabel("Cloud Mass" + r"$~[M_\odot]$", fontsize=16)
plt.ylabel("N", fontsize=16)
plt.title("Distribution of Cloud Masses")
plt.show()
plt.savefig("../figures/massdist.pdf")
