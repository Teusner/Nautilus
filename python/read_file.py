from distutils.command.build import build
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    A = np.load("build/Pressure.npy")

    print(f"Somme : {np.sum(A)}")

    m, M = np.min(A), np.max(A)
    print(f"[{m}, {M}]")
    print(A.shape)

    fig, ax = plt.subplots()
    data = ax.imshow(A[:, :, 2], aspect="equal", cmap="jet", vmin=m, vmax=M)
    fig.colorbar(data, extend='both')
    # plt.show()

    for i in range(A.shape[2]):
        Im = A[:, :, i]
        print(f"{i}: [{np.mean(Im)}, {np.std(Im)}]")
        data = ax.imshow(Im, aspect="equal", cmap="jet", vmin=m, vmax=M)
        plt.pause(1)