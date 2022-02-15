from distutils.command.build import build
import enum
from turtle import color
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    B = np.load("build/Boundary.npy")
    m, M = np.min(B), np.max(B)
    print(f"[{m}, {M}]")


    n = B.shape[0]
    fig, ax = plt.subplots(3, 3)
    L = np.array([2, n//2, n-2])
    for i, l in enumerate(L):
        datax = ax[i, 0].imshow(B[l, :, :], aspect="equal", cmap="jet", vmin=m, vmax=M)
        datay = ax[i, 1].imshow(B[:, l, :], aspect="equal", cmap="jet", vmin=m, vmax=M)
        dataz = ax[i, 2].imshow(B[:, :, l], aspect="equal", cmap="jet", vmin=m, vmax=M)
    plt.colorbar(datax)
    plt.show()
