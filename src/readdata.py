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

os.chdir(os.path.dirname(__file__))

res512, res1024, res2048 = np.load("../data/masses.npy", allow_pickle=True)

res = [512, 1024, 2048]
masses = [res512, res1024, res2048]
colors = ["steelblue", "darkslategray", "lightsteelblue"]

nbins = 100

def cell_vol_kpc(res):
    return (10.0 / float(res)) ** 3

for i, r in enumerate(res):
    plt.figure(figsize=(10, 8))
    _ = plt.hist(np.log10(masses[i] * cell_vol_kpc(r)), bins=100,
                 color=colors[i])

    plt.yscale("log")
    plt.xlabel("Cloud Mass" + r"$~[\log M_\odot]$", fontsize=16)
    plt.ylabel("N", fontsize=16)
    plt.title(f"CGOLS Cloud Masses R={r}")
    plt.savefig(f"../figures/massdist{r}.pdf")
    # plt.show()

plt.figure(figsize=(10, 8))
n2048, bins2048, patches2048 = plt.hist(np.log10(res2048 * cell_vol_kpc(res[2])), bins=100, color=colors[2], label=r"$R={2048}$")
n1024, bins1024, patches1024 = plt.hist(np.log10(res1024 * cell_vol_kpc(res[1])), bins=100, color=colors[1], label=r"$R={1024}$")
n512, bins512, patches512 = plt.hist(np.log10(res512 * cell_vol_kpc(res[0])), bins=100, color=colors[0], label=r"$R={512}$")

n = [n512, n1024, n2048]
bins = [bins512, bins1024, bins2048]
patches = [patches512, patches1024, patches2048]

plt.yscale("log")
plt.xlabel("Cloud Mass" + r"$~[\log M_\odot]$", fontsize=16)
plt.ylabel("N", fontsize=16)
plt.title(f"CGOLS Cloud Masses")
plt.legend()
plt.savefig(f"../figures/massdist.pdf")
# plt.show()

fig = plt.figure(figsize=(10, 8))
for i, r in enumerate(n):
    plt.semilogy(bins[i][1:], bins[i][1:] * n[i], label=f"$R={res[i]}$",
                 color=colors[i])
    plt.legend()
    plt.ylabel("Total Mass" + r"$~[\log M_\odot]$")
    plt.xlabel("Mass" + r"$~[\log M_\odot]$")
    plt.title("CGOLS Cloud Masses")

plt.savefig(f"../figures/totalmassdist.pdf")


plt.show()