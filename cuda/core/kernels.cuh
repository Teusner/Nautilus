#pragma once

#include <core/Material.cuh>
#include <core/Module.cuh>
#include <thrust/random.h>
#define N 10

__constant__ DeviceMaterial M[N];
__constant__ unsigned int l;
__constant__ float dt;


template<unsigned int x, unsigned int y, unsigned int z>
__global__ void F(float t, Emitter* E, float* F);

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Ux(float dt, float* Ux, float* Px, float* Pxy, float* Pxz, unsigned int* S);

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Uy(float dt, float* Uy, float* Py, float* Pxy, float* Pyz, unsigned int* S);

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Uz(float dt, float* Uz, float* Pz, float* Pyz, float* Pxz, unsigned int* S);

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Rxx(float dt, float* Rx, float* Ux, float* Uy, float* Uz, unsigned int* S, float tau_sigma);

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Ryy(float dt, float* Ry, float* Ux, float* Uy, float* Uz, unsigned int* S, float tau_sigma);

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Rzz(float dt, float* Rz, float* Ux, float* Uy, float* Uz, unsigned int* S, float tau_sigma);

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Rxy(float dt, float* Rxy, float* Ux, float* Uy, unsigned int* S, float tau_sigma);

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Ryz(float dt, float* Ryz, float* Uy, float* Uz, unsigned int* S, float tau_sigma);

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Rxz(float dt, float* Rxz, float* Ux, float* Uz, unsigned int* S, float tau_sigma);

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Pxx(float dt, float* Px, float* Ux, float* Uy, float* Uz, float* Rx, unsigned int* S);

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Pyy(float dt, float* Py, float* Ux, float* Uy, float* Uz, float* Rx, unsigned int* S);

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Pzz(float dt, float* Pz, float* Ux, float* Uy, float* Uz, float* Rz, unsigned int* S);

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Pxy(float dt, float* Pxy, float* Ux, float* Uy, float* Rxy, unsigned int* S);

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Pyz(float dt, float* Pyz, float* Uy, float* Uz, float* Ryz, unsigned int* S);

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Pxz(float dt, float* Pxz, float* Ux, float* Uz, float* Rxz, unsigned int* S);


/// Implementation
template<unsigned int x, unsigned int y, unsigned int z, typename T>
__global__ void F(float t, T* E, float* F) {
    unsigned int index = threadIdx.x;
    T e = E[index];
    // printf("%f\n", e(t));
    F[e.x + e.y * x + e.z * y] = e(t);
}

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Ux(float dt, float* Ux, float* Px, float* Pxy, float* Pxz, unsigned int* S) {
    unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned int j = (blockIdx.y * blockDim.y) + threadIdx.y;
    unsigned int k = (blockIdx.z * blockDim.z) + threadIdx.z;

    if (i>=2 && i<x-2 && j>=2 && j<y-2 && k>=2 && k<z-2) {
        float dPxx = - Px[i+2 + j*x + k*y] + 27 * (Px[i+1 + j*x + k*y] - Px[i + j*x + k*y]) + Px[i-1 + j*x + k*y];
        float dPxy = - Pxy[i + (j+1)*x + k*y] + 27 * (Pxy[i + j*x + k*y] - Pxy[i + (j-1)*x + k*y]) + Pxy[i + (j-2)*x + k*y];
        float dPxz = - Pxz[i + j*x + (k+1)*y] + 27 * (Pxz[i + j*x + k*y] - Pxz[i + j*x + (k-1)*y]) + Pxz[i + j*x + (k-2)*y];
        unsigned int material_index = S[i + j*x + k*y];
        Ux[i + j*x + k*y] += dt * M[material_index].inv_rho * (dPxx + dPxy + dPxz);
    }
}

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Uy(float dt, float* Uy, float* Py, float* Pxy, float* Pyz, unsigned int* S) {
    unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned int j = (blockIdx.y * blockDim.y) + threadIdx.y;
    unsigned int k = (blockIdx.z * blockDim.z) + threadIdx.z;

    if (i>=2 && i<x-2 && j>=2 && j<y-2 && k>=2 && k<z-2) {
        float dPyy = - Py[i + (j+2)*x + k*y] + 27 * (Py[i + (j+1)*x + k*y] - Py[i + j*x + k*y]) + Py[i + (j-1)*x + k*y];
        float dPxy = - Pxy[i+1 + j*x + k*y] + 27 * (Pxy[i + j*x + k*y] - Pxy[i-1 + j*x + k*y]) + Pxy[i-2 + j*x + k*y];
        float dPyz = - Pyz[i + j*x + (k+1)*y] + 27 * (Pyz[i + j*x + k*y] - Pyz[i + j*x + (k-1)*y]) + Pyz[i + j*x + (k-2)*y];
        unsigned int material_index = S[i + j*x + k*y];
        Uy[i + j*x + k*y] += dt * M[material_index].inv_rho * (dPyy + dPxy + dPyz);
    }
}

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Uz(float dt, float* Uz, float* Pz, float* Pyz, float* Pxz, unsigned int* S) {
    unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned int j = (blockIdx.y * blockDim.y) + threadIdx.y;
    unsigned int k = (blockIdx.z * blockDim.z) + threadIdx.z;

    if (i>=2 && i<x-2 && j>=2 && j<y-2 && k>=2 && k<z-2) {
        float dPzz = - Pz[i + j*x + (k+2)*y] + 27 * (Pz[i + j*x + (k+1)*y] - Pz[i + j*x + k*y]) + Pz[i + j*x + (k-1)*y];
        float dPyz = - Pyz[i+1 + j*x + k*y] + 27 * (Pyz[i + j*x + k*y] - Pyz[i-1 + j*x + k*y]) + Pyz[i-2 + j*x + k*y];
        float dPxz = - Pxz[i + (j+1)*x + k*y] + 27 * (Pxz[i + j*x + k*y] - Pxz[i + (j-1)*x + k*y]) + Pxz[i + (j-2)*x + k*y];
        unsigned int material_index = S[i + j*x + k*y];
        Uz[i + j*x + k*y] += dt * M[material_index].inv_rho * (dPzz + dPyz + dPxz);
    }
}

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Rxx(float dt, float* Rx, float* Ux, float* Uy, float* Uz, unsigned int* S, float tau_sigma) {
    unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned int j = (blockIdx.y * blockDim.y) + threadIdx.y;
    unsigned int k = (blockIdx.z * blockDim.z) + threadIdx.z;

    // printf("%x", l);

    if (i>=2 && i<x-2 && j>=2 && j<y-2 && k>=2 && k<z-2) {
        float Uxx = - Ux[i+1 + j*x + k*y] + 27 * (Ux[i + j*x + k*y] - Ux[i-1 + j*x + k*y]) + Ux[i-2 + j*x + k*y];
        float Uyy = - Uy[i + (j+1)*x + k*y] + 27 * (Uy[i + j*x + k*y] - Uy[i + (j-1)*x + k*y]) + Uy[i + (j-2)*x + k*y];
        float Uzz = - Uz[i + j*x + (k+1)*y] + 27 * (Uz[i + j*x + k*y] - Uz[i + j*x + (k-1)*y]) + Uz[i + j*x + (k-2)*y];
        unsigned int material_index = S[i + j*x + k*y];
        float a = (2 * tau_sigma - dt) / (2 * tau_sigma + dt);
        float b = 2 * dt / (2 * tau_sigma + dt);
        Rx[i + j*x + k*y] += a * Rx[i + j*x + k*y] - b * M[material_index].eta_tau_gamma_p * (Uxx + Uyy + Uzz) - 2 * M[material_index].mu_tau_gamma_s * (Uyy + Uzz);
    }
}

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Ryy(float dt, float* Ry, float* Ux, float* Uy, float* Uz, unsigned int* S, float tau_sigma) {
    unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned int j = (blockIdx.y * blockDim.y) + threadIdx.y;
    unsigned int k = (blockIdx.z * blockDim.z) + threadIdx.z;

    if (i>=2 && i<x-2 && j>=2 && j<y-2 && k>=2 && k<z-2) {
        float Uxx = - Ux[i+1 + j*x + k*y] + 27 * (Ux[i + j*x + k*y] - Ux[i-1 + j*x + k*y]) + Ux[i-2 + j*x + k*y];
        float Uyy = - Uy[i + (j+1)*x + k*y] + 27 * (Uy[i + j*x + k*y] - Uy[i + (j-1)*x + k*y]) + Uy[i + (j-2)*x + k*y];
        float Uzz = - Uz[i + j*x + (k+1)*y] + 27 * (Uz[i + j*x + k*y] - Uz[i + j*x + (k-1)*y]) + Uz[i + j*x + (k-2)*y];
        unsigned int material_index = S[i + j*x + k*y];
        float a = (2 * tau_sigma - dt) / (2 * tau_sigma + dt);
        float b = 2 * dt / (2 * tau_sigma + dt);
        Ry[i + j*x + k*y] += a * Ry[i + j*x + k*y] - b * M[material_index].eta_tau_gamma_p * (Uxx + Uyy + Uzz) - 2 * M[material_index].mu_tau_gamma_s * (Uxx + Uzz);
    }
}

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Rzz(float dt, float* Rz, float* Ux, float* Uy, float* Uz, unsigned int* S, float tau_sigma) {
    unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned int j = (blockIdx.y * blockDim.y) + threadIdx.y;
    unsigned int k = (blockIdx.z * blockDim.z) + threadIdx.z;

    if (i>=2 && i<x-2 && j>=2 && j<y-2 && k>=2 && k<z-2) {
        float Uxx = - Ux[i+1 + j*x + k*y] + 27 * (Ux[i + j*x + k*y] - Ux[i-1 + j*x + k*y]) + Ux[i-2 + j*x + k*y];
        float Uyy = - Uy[i + (j+1)*x + k*y] + 27 * (Uy[i + j*x + k*y] - Uy[i + (j-1)*x + k*y]) + Uy[i + (j-2)*x + k*y];
        float Uzz = - Uz[i + j*x + (k+1)*y] + 27 * (Uz[i + j*x + k*y] - Uz[i + j*x + (k-1)*y]) + Uz[i + j*x + (k-2)*y];
        unsigned int material_index = S[i + j*x + k*y];
        float a = (2 * tau_sigma - dt) / (2 * tau_sigma + dt);
        float b = 2 * dt / (2 * tau_sigma + dt);
        Rz[i + j*x + k*y] += a * Rz[i + j*x + k*y] - b * M[material_index].eta_tau_gamma_p * (Uxx + Uyy + Uzz) - 2 * M[material_index].mu_tau_gamma_s * (Uxx + Uyy);
    }
}

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Rxy(float dt, float* Rxy, float* Ux, float* Uy, unsigned int* S, float tau_sigma) {
    unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned int j = (blockIdx.y * blockDim.y) + threadIdx.y;
    unsigned int k = (blockIdx.z * blockDim.z) + threadIdx.z;

    if (i>=2 && i<x-2 && j>=2 && j<y-2 && k>=2 && k<z-2) {
        float Uxy = - Ux[i + (j+2)*x + k*y] + 27 * (Ux[i + (j+1)*x + k*y] - Ux[i + j*x + k*y]) + Ux[i + (j-1)*x + k*y];
        float Uyx = - Uy[i+2 + j*x + k*y] + 27 * (Uy[i+1 + j*x + k*y] - Uy[i + j*x + k*y]) + Uy[i-1 + j*x + k*y];
        unsigned int material_index = S[i + j*x + k*y];
        float a = (2 * tau_sigma - dt) / (2 * tau_sigma + dt);
        float b = 2 * dt / (2 * tau_sigma + dt);
        Rxy[i + j*x + k*y] += a * Rxy[i + j*x + k*y] - b * M[material_index].mu_tau_gamma_s * (Uxy + Uyx);
    }
}

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Ryz(float dt, float* Ryz, float* Uy, float* Uz, unsigned int* S, float tau_sigma) {
    unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned int j = (blockIdx.y * blockDim.y) + threadIdx.y;
    unsigned int k = (blockIdx.z * blockDim.z) + threadIdx.z;

    if (i>=2 && i<x-2 && j>=2 && j<y-2 && k>=2 && k<z-2) {
        float Uyz = - Uy[i + j*x + (k+2)*y] + 27 * (Uy[i + j*x + (k+1)*y] - Uy[i + j*x + k*y]) + Uy[i + j*x + (k-1)*y];
        float Uzy = - Uz[i + (j+2)*x + k*y] + 27 * (Uz[i + (j+1)*x + k*y] - Uz[i + j*x + k*y]) + Uz[i + (j-1)*x + k*y];
        unsigned int material_index = S[i + j*x + k*y];
        float a = (2 * tau_sigma - dt) / (2 * tau_sigma + dt);
        float b = 2 * dt / (2 * tau_sigma + dt);
        Ryz[i + j*x + k*y] += a * Ryz[i + j*x + k*y] - b * M[material_index].mu_tau_gamma_s * (Uyz + Uzy);
    }
}

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Rxz(float dt, float* Rxz, float* Ux, float* Uz, unsigned int* S, float tau_sigma) {
    unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned int j = (blockIdx.y * blockDim.y) + threadIdx.y;
    unsigned int k = (blockIdx.z * blockDim.z) + threadIdx.z;

    if (i>=2 && i<x-2 && j>=2 && j<y-2 && k>=2 && k<z-2) {
        float Uxz = - Ux[i + j*x + (k+2)*y] + 27 * (Ux[i + j*x + (k+1)*y] - Ux[i + j*x + k*y]) + Ux[i + j*x + (k-1)*y];
        float Uzx = - Uz[i+2 + j*x + k*y] + 27 * (Uz[i+1 + j*x + k*y] - Uz[i + j*x + k*y]) + Uz[i-1 + j*x + k*y];
        unsigned int material_index = S[i + j*x + k*y];
        float a = (2 * tau_sigma - dt) / (2 * tau_sigma + dt);
        float b = 2 * dt / (2 * tau_sigma + dt);
        Rxz[i + j*x + k*y] += a * Rxz[i + j*x + k*y] - b * M[material_index].mu_tau_gamma_s * (Uxz + Uzx);
    }
}

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Pxx(float dt, float* Px, float* Ux, float* Uy, float* Uz, float* Rx, unsigned int* S, float* F) {
    unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned int j = (blockIdx.y * blockDim.y) + threadIdx.y;
    unsigned int k = (blockIdx.z * blockDim.z) + threadIdx.z;

    if (i>=2 && i<x-2 && j>=2 && j<y-2 && k>=2 && k<z-2) {
        float Uxx = - Ux[i+1 + j*x + k*y] + 27 * (Ux[i + j*x + k*y] - Ux[i-1 + j*x + k*y]) + Ux[i-2 + j*x + k*y];
        float Uyy = - Uy[i + (j+1)*x + k*y] + 27 * (Uy[i + j*x + k*y] - Uy[i + (j-1)*x + k*y]) + Uy[i + (j-2)*x + k*y];
        float Uzz = - Uz[i + j*x + (k+1)*y] + 27 * (Uz[i + j*x + k*y] - Uz[i + j*x + (k-1)*y]) + Uz[i + j*x + (k-2)*y];
        unsigned int material_index = S[i + j*x + k*y];
        Px[i + j*x + k*y] += dt * M[material_index].eta_tau_gamma_p * (Uxx + Uyy + Uzz) - 2 * M[material_index].mu_tau_gamma_s * (Uyy + Uzz) + Rx[i + j*x + k*y] + F[i + j*x + k*y];
    }
}

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Pyy(float dt, float* Py, float* Ux, float* Uy, float* Uz, float* Ry, unsigned int* S, float* F) {
    unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned int j = (blockIdx.y * blockDim.y) + threadIdx.y;
    unsigned int k = (blockIdx.z * blockDim.z) + threadIdx.z;

    if (i>=2 && i<x-2 && j>=2 && j<y-2 && k>=2 && k<z-2) {
        float Uxx = - Ux[i+1 + j*x + k*y] + 27 * (Ux[i + j*x + k*y] - Ux[i-1 + j*x + k*y]) + Ux[i-2 + j*x + k*y];
        float Uyy = - Uy[i + (j+1)*x + k*y] + 27 * (Uy[i + j*x + k*y] - Uy[i + (j-1)*x + k*y]) + Uy[i + (j-2)*x + k*y];
        float Uzz = - Uz[i + j*x + (k+1)*y] + 27 * (Uz[i + j*x + k*y] - Uz[i + j*x + (k-1)*y]) + Uz[i + j*x + (k-2)*y];
        unsigned int material_index = S[i + j*x + k*y];
        Py[i + j*x + k*y] += dt * M[material_index].eta_tau_gamma_p * (Uxx + Uyy + Uzz) - 2 * M[material_index].mu_tau_gamma_s * (Uxx + Uzz) + Ry[i + j*x + k*y] + F[i + j*x + k*y];
    }
}

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Pzz(float dt, float* Pz, float* Ux, float* Uy, float* Uz, float* Rz, unsigned int* S, float* F) {
    unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned int j = (blockIdx.y * blockDim.y) + threadIdx.y;
    unsigned int k = (blockIdx.z * blockDim.z) + threadIdx.z;

    if (i>=2 && i<x-2 && j>=2 && j<y-2 && k>=2 && k<z-2) {
        float Uxx = - Ux[i+1 + j*x + k*y] + 27 * (Ux[i + j*x + k*y] - Ux[i-1 + j*x + k*y]) + Ux[i-2 + j*x + k*y];
        float Uyy = - Uy[i + (j+1)*x + k*y] + 27 * (Uy[i + j*x + k*y] - Uy[i + (j-1)*x + k*y]) + Uy[i + (j-2)*x + k*y];
        float Uzz = - Uz[i + j*x + (k+1)*y] + 27 * (Uz[i + j*x + k*y] - Uz[i + j*x + (k-1)*y]) + Uz[i + j*x + (k-2)*y];
        unsigned int material_index = S[i + j*x + k*y];
        Pz[i + j*x + k*y] += dt * M[material_index].eta_tau_gamma_p * (Uxx + Uyy + Uzz) - 2 * M[material_index].mu_tau_gamma_s * (Uxx + Uyy) + Rz[i + j*x + k*y] + F[i + j*x + k*y];
    }
}

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Pxy(float dt, float* Pxy, float* Ux, float* Uy, float* Rxy, unsigned int* S) {
    unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned int j = (blockIdx.y * blockDim.y) + threadIdx.y;
    unsigned int k = (blockIdx.z * blockDim.z) + threadIdx.z;

    if (i>=2 && i<x-2 && j>=2 && j<y-2 && k>=2 && k<z-2) {
        float Uxy = - Ux[i + (j+2)*x + k*y] + 27 * (Ux[i + (j+1)*x + k*y] - Ux[i + j*x + k*y]) + Ux[i + (j-1)*x + k*y];
        float Uyx = - Uy[i+2 + j*x + k*y] + 27 * (Uy[i+1 + j*x + k*y] - Uy[i + j*x + k*y]) + Uy[i-1 + j*x + k*y];
        unsigned int material_index = S[i + j*x + k*y];
        Pxy[i + j*x + k*y] += dt * M[material_index].mu_tau_gamma_s * (Uxy + Uyx) + Rxy[i + j*x + k*y];
    }
}

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Pyz(float dt, float* Pyz, float* Uy, float* Uz, float* Ryz, unsigned int* S) {
    unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned int j = (blockIdx.y * blockDim.y) + threadIdx.y;
    unsigned int k = (blockIdx.z * blockDim.z) + threadIdx.z;

    if (i>=2 && i<x-2 && j>=2 && j<y-2 && k>=2 && k<z-2) {
        float Uyz = - Uy[i + j*x + (k+2)*y] + 27 * (Uy[i + j*x + (k+1)*y] - Uy[i + j*x + k*y]) + Uy[i + j*x + (k-1)*y];
        float Uzy = - Uz[i + (j+2)*x + k*y] + 27 * (Uz[i + (j+1)*x + k*y] - Uz[i + j*x + k*y]) + Uz[i + (j-1)*x + k*y];
        unsigned int material_index = S[i + j*x + k*y];
        Pyz[i + j*x + k*y] += dt * M[material_index].mu_tau_gamma_s * (Uyz + Uzy) + Ryz[i + j*x + k*y];
    }
}

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Pxz(float dt, float* Pxz, float* Ux, float* Uz, float* Rxz, unsigned int* S) {
    unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned int j = (blockIdx.y * blockDim.y) + threadIdx.y;
    unsigned int k = (blockIdx.z * blockDim.z) + threadIdx.z;

    if (i>=2 && i<x-2 && j>=2 && j<y-2 && k>=2 && k<z-2) {
        float Uxz = - Ux[i + j*x + (k+2)*y] + 27 * (Ux[i + j*x + (k+1)*y] - Ux[i + j*x + k*y]) + Ux[i + j*x + (k-1)*y];
        float Uzx = - Uz[i+2 + j*x + k*y] + 27 * (Uz[i+1 + j*x + k*y] - Uz[i + j*x + k*y]) + Uz[i-1 + j*x + k*y];
        unsigned int material_index = S[i + j*x + k*y];
        Pxz[i + j*x + k*y] += dt * M[material_index].mu_tau_gamma_s * (Uxz + Uzx) + Rxz[i + j*x + k*y];
    }
}

struct prg {
    float a, b;

    __host__ __device__
    prg(float _a=0.f, float _b=1.f) : a(_a), b(_b) {};

    __host__ __device__
        float operator()(const unsigned int n) const {
            thrust::default_random_engine rng;
            thrust::uniform_real_distribution<float> dist(a, b);
            rng.discard(n);

            return dist(rng);
        }
};