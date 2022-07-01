import numpy as np
from mayavi import mlab

if __name__ == "__main__":
    s = np.sum(np.load("build/Pressure.npy"), axis=3)

    print(s.shape)

    src = mlab.pipeline.scalar_field(s)

    n = 6
    alpha = np.linspace(0.4, 0.9, n)
    for k in alpha:
        mlab.pipeline.iso_surface(src, contours=[s.min()+k*s.ptp(), ], opacity=k**5)
    mlab.show()
