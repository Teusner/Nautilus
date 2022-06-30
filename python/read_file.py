import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    A = np.load("build/Pressure.npy")

    print(f"Somme : {np.sum(A)}")

    m, M = np.min(A), np.max(A)
    print(f"[{m}, {M}]")
    print(A.shape)

    fig, ax = plt.subplots()
    data = ax.imshow(np.sum(A[:, :, :], axis=2), aspect="equal", cmap="jet", vmin=m, vmax=M)
    fig.colorbar(data, extend='both')
    # plt.show()

    for i in range(A.shape[2]):
        Im = np.sum(A[:, :, i], axis=2)
        print(f"{i}: [{np.mean(Im)}, {np.std(Im)}]")
        data = ax.imshow(Im, aspect="equal", cmap="jet", vmin=m, vmax=M)
        plt.pause(0.3)
