from distutils.command.build import build
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    A = np.load("build/Pressure.npy")

    for i, Im in enumerate(A):
        print(f"{i}: [{np.mean(Im)}, {np.std(Im)}]")
        plt.imshow(Im, aspect="equal", cmap="jet")
        plt.pause(0.1)