import numpy as np
import matplotlib.pyplot as plt

if __name__ =="__main__":
    f = np.arange(2, 25.1, 1)
    omega = 2*np.pi*f

    k = 1
    y = [f[0]]
    while y[-1]*2**k <= f[-1] :
        y.append(y[-1]*2**k)
        k+=1

    z = []
    for k in range(0, int(np.log2(f[-1]))-1):
        z.append(f[k]*2**k)

    y = np.asarray(y)
    z = np.asarray(z)
    print(y)
    print(f"{1e3/(2*np.pi*y)} ms")

    print(z)
    print(f"{1e3/(2*np.pi*z)} ms")
