

__global__ Ux(float* Ux, float* Px, float* Pxy, float* Pxz, float dt, float* rho, dim3 d) {

    unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned int j = (blockIdx.y * blockDim.y) + threadIdx.y;
    unsigned int k = (blockIdx.z * blockDim.z) + threadIdx.z;

    float dPxx = - Px[i+2 + j*d.x + k*d.y] + 27 * (Px[i+1 + j*d.x + k*d.y] - Px[i + j*d.x + k*d.y]) + Px[i-1 + j*d.x + k*d.y];
    float dPxy = - Pxy[i + (j+1)*d.x + k*d.y] + 27 * (Pxy[i + j*d.x + k*d.y] - Pxy[i + (j-1)*d.x + k * d.y]) + Pxy[i + (j-2)*d.x + k *d.y];
    float dPxz = - Pxz[i + j*d.x + (k+1)*d.y] + 27 * (Pxz[i + j*d.x + k*d.y] - Pxz[i j*d.x + (k-1)*d.y]) + Pxz[i + j*d.x + (k-2)*d.y];
    Ux[i + j*d.x + k*d.y] += dt / rho[i + j*d.x + k*d.y] * (dPxx + dPxy + dPxz);
}

__global__ Uy(float* Uy, float* Py, float* Pxy, float* Pyz, float dt, float* rho, dim3 d) {
    unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned int j = (blockIdx.y * blockDim.y) + threadIdx.y;
    unsigned int k = (blockIdx.z * blockDim.z) + threadIdx.z;

    float dPyy = - Py[i + (j+2)*d.x + k*d.y] + 27 * (Py[i + (j+1)*d.x + k*d.y] - Py[i + j*d.x + k*d.y]) + Py[i + (j-1)*d.x + k*d.y];
    float dPxy = - Pxy[i+1 + j*d.x + k*d.y] + 27 * (Pxy[i + j*d.x + k*d.y] - Pxy[i-1 + j*d.x + k*d.y]) + Pxy[i-2 + j*d.x + k*d.y];
    float dPyz = - Pyz[i + j*d.x + (k+1)*d.y] + 27 * (Pyz[i + j*d.x + k*d.y] - Pyz[i + j*d.x + (k-1)*d.y]) + Pyz[i + j*d.x + (k-2)*d.y];
    Uy[i + j*d.x + k*d.y] += dt / rho[i + j*d.x + k*d.y] * (dPyy + dPxy + dPyz);
}

__global__ Uz(float* Uz, float* Pz, float* Pyz, float* Pxz, float dt, float* rho, dim3 d) {
    unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned int j = (blockIdx.y * blockDim.y) + threadIdx.y;
    unsigned int k = (blockIdx.z * blockDim.z) + threadIdx.z;

    float dPzz = - Pz[i + j*d.x + (k+2)*d.y] + 27 * (Py[i + j*d.x + (k+1)*d.y] - Py[i + j*d.x + k*d.y]) + Py[i + j*d.x + (k-1)*d.y];
    float dPxy = - Pyz[i+1 + j*d.x + k*d.y] + 27 * (Pyz[i + j*d.x + k*d.y] - Pyz[i-1 + j*d.x + k*d.y]) + Pyz[i-2 + j*d.x + k*d.y];
    float dPyz = - Pxz[i + (j+1)*d.x + k*d.y] + 27 * (Pxz[i + j*d.x + k*d.y] - Pxz[i + (j-1)*d.x + k*d.y]) + Pxz[i + (j-2)*d.x + k*d.y];
    Uz[i + j*d.x + k*d.y] += dt / rho[i + j*d.x + k*d.y] * (dPzz + dPyz + dPxz);
}