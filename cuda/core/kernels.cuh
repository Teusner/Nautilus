#pragma once

#include <core/Material.cuh>
#include <core/Module.cuh>
#include <thrust/random.h>
#define N 10

__constant__ DeviceMaterial M[N];
__constant__ unsigned int d_l;
__constant__ float d_dt;
__constant__ float d_tau_sigma[N];
__constant__ float d_alpha[3];


template<unsigned int x, unsigned int y, unsigned int z>
__global__ void F(float t, Emitter* E, float* F);

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Ux(float* Ux, float* Px, float* Pxy, float* Pxz, unsigned int* S);

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Uy(float* Uy, float* Py, float* Pxy, float* Pyz, unsigned int* S);

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Uz(float* Uz, float* Pz, float* Pyz, float* Pxz, unsigned int* S);

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Uxx(float* Ux, float* Uxx);

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Uyy(float* Uy, float* Uyy);

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Uzz(float* Uz, float* Uzz);

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Rxx(float* Rx, float* Ux, float* Uy, float* Uz, unsigned int* S);

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Ryy(float* Ry, float* Ux, float* Uy, float* Uz, unsigned int* S);

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Rzz(float* Rz, float* Ux, float* Uy, float* Uz, unsigned int* S);

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Rxy(float* Rxy, float* Ux, float* Uy, unsigned int* S);

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Ryz(float* Ryz, float* Uy, float* Uz, unsigned int* S);

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Rxz(float* Rxz, float* Ux, float* Uz, unsigned int* S);

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Pxx(float* Px, float* Ux, float* Uy, float* Uz, float* Rx, unsigned int* S);

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Pyy(float* Py, float* Ux, float* Uy, float* Uz, float* Rx, unsigned int* S);

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Pzz(float* Pz, float* Ux, float* Uy, float* Uz, float* Rz, unsigned int* S);

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Pxy(float* Pxy, float* Ux, float* Uy, float* Rxy, unsigned int* S);

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Pyz(float* Pyz, float* Uy, float* Uz, float* Ryz, unsigned int* S);

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Pxz(float* Pxz, float* Ux, float* Uz, float* Rxz, unsigned int* S);


/// Implementation
template<unsigned int x, unsigned int y, unsigned int z, typename T>
__global__ void F(float t, T* E, float* F) {
    unsigned int index = threadIdx.x;
    T e = E[index];
    // printf("%f\n", e(t));
    F[e.x + e.y * x + e.z * y] = e(t);
}

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Ux(float* Ux, float* Px, float* Pxy, float* Pxz, unsigned int* S) {
    unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned int j = (blockIdx.y * blockDim.y) + threadIdx.y;
    unsigned int k = (blockIdx.z * blockDim.z) + threadIdx.z;

    if (i>=2 && i<x-2 && j>=2 && j<y-2 && k>=2 && k<z-2) {
        float dPxx = (- Px[i+2 + j*x + k*y] + 27 * (Px[i+1 + j*x + k*y] - Px[i + j*x + k*y]) + Px[i-1 + j*x + k*y]) * d_alpha[0];
        float dPxy = (- Pxy[i + (j+1)*x + k*y] + 27 * (Pxy[i + j*x + k*y] - Pxy[i + (j-1)*x + k*y]) + Pxy[i + (j-2)*x + k*y]) * d_alpha[1];
        float dPxz = (- Pxz[i + j*x + (k+1)*y] + 27 * (Pxz[i + j*x + k*y] - Pxz[i + j*x + (k-1)*y]) + Pxz[i + j*x + (k-2)*y]) * d_alpha[2];
        unsigned int material_index = S[i + j*x + k*y];
        Ux[i + j*x + k*y] += d_dt * M[material_index].inv_rho * (dPxx + dPxy + dPxz);
    }
}

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Uy(float* Uy, float* Py, float* Pxy, float* Pyz, unsigned int* S) {
    unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned int j = (blockIdx.y * blockDim.y) + threadIdx.y;
    unsigned int k = (blockIdx.z * blockDim.z) + threadIdx.z;

    if (i>=2 && i<x-2 && j>=2 && j<y-2 && k>=2 && k<z-2) {
        float dPyy = (- Py[i + (j+2)*x + k*y] + 27 * (Py[i + (j+1)*x + k*y] - Py[i + j*x + k*y]) + Py[i + (j-1)*x + k*y]) * d_alpha[0];
        float dPxy = (- Pxy[i+1 + j*x + k*y] + 27 * (Pxy[i + j*x + k*y] - Pxy[i-1 + j*x + k*y]) + Pxy[i-2 + j*x + k*y]) * d_alpha[1];
        float dPyz = (- Pyz[i + j*x + (k+1)*y] + 27 * (Pyz[i + j*x + k*y] - Pyz[i + j*x + (k-1)*y]) + Pyz[i + j*x + (k-2)*y]) * d_alpha[2];
        unsigned int material_index = S[i + j*x + k*y];
        Uy[i + j*x + k*y] += d_dt * M[material_index].inv_rho * (dPyy + dPxy + dPyz);
    }
}

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Uz(float* Uz, float* Pz, float* Pyz, float* Pxz, unsigned int* S) {
    unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned int j = (blockIdx.y * blockDim.y) + threadIdx.y;
    unsigned int k = (blockIdx.z * blockDim.z) + threadIdx.z;

    if (i>=2 && i<x-2 && j>=2 && j<y-2 && k>=2 && k<z-2) {
        float dPzz = (- Pz[i + j*x + (k+2)*y] + 27 * (Pz[i + j*x + (k+1)*y] - Pz[i + j*x + k*y]) + Pz[i + j*x + (k-1)*y]) * d_alpha[0];
        float dPyz = (- Pyz[i+1 + j*x + k*y] + 27 * (Pyz[i + j*x + k*y] - Pyz[i-1 + j*x + k*y]) + Pyz[i-2 + j*x + k*y]) * d_alpha[1];
        float dPxz = (- Pxz[i + (j+1)*x + k*y] + 27 * (Pxz[i + j*x + k*y] - Pxz[i + (j-1)*x + k*y]) + Pxz[i + (j-2)*x + k*y]) * d_alpha[2];
        unsigned int material_index = S[i + j*x + k*y];
        Uz[i + j*x + k*y] += d_dt * M[material_index].inv_rho * (dPzz + dPyz + dPxz);
    }
}

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Uxx(float* Ux, float* Uxx) {
    unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned int j = (blockIdx.y * blockDim.y) + threadIdx.y;
    unsigned int k = (blockIdx.z * blockDim.z) + threadIdx.z;

    if (i>=2 && i<x-2 && j>=2 && j<y-2 && k>=2 && k<z-2) {
        Uxx[i + j*x + k*y] = (- Ux[i+1 + j*x + k*y] + 27 * (Ux[i + j*x + k*y] - Ux[i-1 + j*x + k*y]) + Ux[i-2 + j*x + k*y]) * d_alpha[0];
    }
}

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Uyy(float* Uy, float* Uyy) {
    unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned int j = (blockIdx.y * blockDim.y) + threadIdx.y;
    unsigned int k = (blockIdx.z * blockDim.z) + threadIdx.z;

    if (i>=2 && i<x-2 && j>=2 && j<y-2 && k>=2 && k<z-2) {
        Uyy[i + j*x + k*y] = (- Uy[i + (j+1)*x + k*y] + 27 * (Uy[i + j*x + k*y] - Uy[i + (j-1)*x + k*y]) + Uy[i + (j-2)*x + k*y]) * d_alpha[1];
    }
}

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Uzz(float* Uz, float* Uzz) {
    unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned int j = (blockIdx.y * blockDim.y) + threadIdx.y;
    unsigned int k = (blockIdx.z * blockDim.z) + threadIdx.z;

    if (i>=2 && i<x-2 && j>=2 && j<y-2 && k>=2 && k<z-2) {
        Uzz[i + j*x + k*y] = (- Uz[i + j*x + (k+1)*y] + 27 * (Uz[i + j*x + k*y] - Uz[i + j*x + (k-1)*y]) + Uz[i + j*x + (k-2)*y]) * d_alpha[2];
    }
}

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Rxx(float* Rx, float* Uxx, float* Uyy, float* Uzz, unsigned int* S) {
    unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned int j = (blockIdx.y * blockDim.y) + threadIdx.y;
    unsigned int k = (blockIdx.z * blockDim.z) + threadIdx.z;

    if (i>=2 && i<x-2 && j>=2 && j<y-2 && k>=2 && k<z-2) {
        unsigned int material_index = S[i + j*x + k*y];
        for (unsigned int q=0; q < d_l; ++q) {
            float a = (2 * d_tau_sigma[q] - d_dt) / (2 * d_tau_sigma[q] + d_dt);
            float b = 2 * d_dt / (2 * d_tau_sigma[q] + d_dt);
            Rx[i + j*x + k*y + q*z] += a * Rx[i + j*x + k*y + q*z] - b * M[material_index].eta_tau_gamma_p * (Uxx[i + j*x + k*y] + Uyy[i + j*x + k*y] + Uzz[i + j*x + k*y]) - 2 * M[material_index].mu_tau_gamma_s * (Uyy[i + j*x + k*y] + Uzz[i + j*x + k*y]);
        }
    }
}

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Ryy(float* Ry, float* Uxx, float* Uyy, float* Uzz, unsigned int* S) {
    unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned int j = (blockIdx.y * blockDim.y) + threadIdx.y;
    unsigned int k = (blockIdx.z * blockDim.z) + threadIdx.z;

    if (i>=2 && i<x-2 && j>=2 && j<y-2 && k>=2 && k<z-2) {
        unsigned int material_index = S[i + j*x + k*y];
        for (unsigned int q=0; q < d_l; ++q) {
            float a = (2 * d_tau_sigma[q] - d_dt) / (2 * d_tau_sigma[q] + d_dt);
            float b = 2 * d_dt / (2 * d_tau_sigma[q] + d_dt);
            Ry[i + j*x + k*y + q*z] += a * Ry[i + j*x + k*y + q*z] - b * M[material_index].eta_tau_gamma_p * (Uxx[i + j*x + k*y] + Uyy[i + j*x + k*y] + Uzz[i + j*x + k*y]) - 2 * M[material_index].mu_tau_gamma_s * (Uxx[i + j*x + k*y] + Uzz[i + j*x + k*y]);
        }
    }
}

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Rzz(float* Rz, float* Uxx, float* Uyy, float* Uzz, unsigned int* S) {
    unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned int j = (blockIdx.y * blockDim.y) + threadIdx.y;
    unsigned int k = (blockIdx.z * blockDim.z) + threadIdx.z;

    if (i>=2 && i<x-2 && j>=2 && j<y-2 && k>=2 && k<z-2) {
        unsigned int material_index = S[i + j*x + k*y];
        for (unsigned int q=0; q < d_l; ++q) {
            float a = (2 * d_tau_sigma[q] - d_dt) / (2 * d_tau_sigma[q] + d_dt);
            float b = 2 * d_dt / (2 * d_tau_sigma[q] + d_dt);
            Rz[i + j*x + k*y + q*z] += a * Rz[i + j*x + k*y + q*z] - b * M[material_index].eta_tau_gamma_p * (Uxx[i + j*x + k*y] + Uyy[i + j*x + k*y] + Uzz[i + j*x + k*y]) - 2 * M[material_index].mu_tau_gamma_s * (Uxx[i + j*x + k*y] + Uyy[i + j*x + k*y]);
        }
    }
}

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Rxy(float* Rxy, float* Ux, float* Uy, unsigned int* S) {
    unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned int j = (blockIdx.y * blockDim.y) + threadIdx.y;
    unsigned int k = (blockIdx.z * blockDim.z) + threadIdx.z;

    if (i>=2 && i<x-2 && j>=2 && j<y-2 && k>=2 && k<z-2) {
        float Uxy = - Ux[i + (j+2)*x + k*y] + 27 * (Ux[i + (j+1)*x + k*y] - Ux[i + j*x + k*y]) + Ux[i + (j-1)*x + k*y];
        float Uyx = - Uy[i+2 + j*x + k*y] + 27 * (Uy[i+1 + j*x + k*y] - Uy[i + j*x + k*y]) + Uy[i-1 + j*x + k*y];
        unsigned int material_index = S[i + j*x + k*y];
        for (unsigned int q=0; q < d_l; ++q) {
            float a = (2 * d_tau_sigma[q] - d_dt) / (2 * d_tau_sigma[q] + d_dt);
            float b = 2 * d_dt / (2 * d_tau_sigma[q] + d_dt);
            Rxy[i + j*x + k*y + q*z] += a * Rxy[i + j*x + k*y + q*z] - b * M[material_index].mu_tau_gamma_s * (Uxy + Uyx);
        }
    }
}

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Ryz(float* Ryz, float* Uy, float* Uz, unsigned int* S) {
    unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned int j = (blockIdx.y * blockDim.y) + threadIdx.y;
    unsigned int k = (blockIdx.z * blockDim.z) + threadIdx.z;

    if (i>=2 && i<x-2 && j>=2 && j<y-2 && k>=2 && k<z-2) {
        float Uyz = - Uy[i + j*x + (k+2)*y] + 27 * (Uy[i + j*x + (k+1)*y] - Uy[i + j*x + k*y]) + Uy[i + j*x + (k-1)*y];
        float Uzy = - Uz[i + (j+2)*x + k*y] + 27 * (Uz[i + (j+1)*x + k*y] - Uz[i + j*x + k*y]) + Uz[i + (j-1)*x + k*y];
        unsigned int material_index = S[i + j*x + k*y];
        for (unsigned int q=0; q < d_l; ++q) {
            float a = (2 * d_tau_sigma[q] - d_dt) / (2 * d_tau_sigma[q] + d_dt);
            float b = 2 * d_dt / (2 * d_tau_sigma[q] + d_dt);
            Ryz[i + j*x + k*y + q*z] += a * Ryz[i + j*x + k*y + q*z] - b * M[material_index].mu_tau_gamma_s * (Uyz + Uzy);
        }
    }
}

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Rxz(float* Rxz, float* Ux, float* Uz, unsigned int* S) {
    unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned int j = (blockIdx.y * blockDim.y) + threadIdx.y;
    unsigned int k = (blockIdx.z * blockDim.z) + threadIdx.z;

    if (i>=2 && i<x-2 && j>=2 && j<y-2 && k>=2 && k<z-2) {
        float Uxz = - Ux[i + j*x + (k+2)*y] + 27 * (Ux[i + j*x + (k+1)*y] - Ux[i + j*x + k*y]) + Ux[i + j*x + (k-1)*y];
        float Uzx = - Uz[i+2 + j*x + k*y] + 27 * (Uz[i+1 + j*x + k*y] - Uz[i + j*x + k*y]) + Uz[i-1 + j*x + k*y];
        unsigned int material_index = S[i + j*x + k*y];
        for (unsigned int q=0; q < d_l; ++q) {
            float a = (2 * d_tau_sigma[q] - d_dt) / (2 * d_tau_sigma[q] + d_dt);
            float b = 2 * d_dt / (2 * d_tau_sigma[q] + d_dt);
            Rxz[i + j*x + k*y + q*z] += a * Rxz[i + j*x + k*y + q*z] - b * M[material_index].mu_tau_gamma_s * (Uxz + Uzx);
        }
    }
}

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Pxx(float* Px, float* Uxx, float* Uyy, float* Uzz, float* Rx, unsigned int* S, float* F) {
    unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned int j = (blockIdx.y * blockDim.y) + threadIdx.y;
    unsigned int k = (blockIdx.z * blockDim.z) + threadIdx.z;

    if (i>=2 && i<x-2 && j>=2 && j<y-2 && k>=2 && k<z-2) {
        unsigned int material_index = S[i + j*x + k*y];
        Px[i + j*x + k*y] += d_dt * M[material_index].eta_tau_gamma_p * (Uxx[i + j*x + k*y] + Uyy[i + j*x + k*y] + Uzz[i + j*x + k*y]) - 2 * M[material_index].mu_tau_gamma_s * (Uyy[i + j*x + k*y] + Uzz[i + j*x + k*y]) + Rx[i + j*x + k*y] + F[i + j*x + k*y];
    }
}

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Pyy(float* Py, float* Uxx, float* Uyy, float* Uzz, float* Ry, unsigned int* S, float* F) {
    unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned int j = (blockIdx.y * blockDim.y) + threadIdx.y;
    unsigned int k = (blockIdx.z * blockDim.z) + threadIdx.z;

    if (i>=2 && i<x-2 && j>=2 && j<y-2 && k>=2 && k<z-2) {
        unsigned int material_index = S[i + j*x + k*y];
        Py[i + j*x + k*y] += d_dt * M[material_index].eta_tau_gamma_p * (Uxx[i + j*x + k*y] + Uyy[i + j*x + k*y] + Uzz[i + j*x + k*y]) - 2 * M[material_index].mu_tau_gamma_s * (Uxx[i + j*x + k*y] + Uzz[i + j*x + k*y]) + Ry[i + j*x + k*y] + F[i + j*x + k*y];
    }
}

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Pzz(float* Pz, float* Uxx, float* Uyy, float* Uzz, float* Rz, unsigned int* S, float* F) {
    unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned int j = (blockIdx.y * blockDim.y) + threadIdx.y;
    unsigned int k = (blockIdx.z * blockDim.z) + threadIdx.z;

    if (i>=2 && i<x-2 && j>=2 && j<y-2 && k>=2 && k<z-2) {
        unsigned int material_index = S[i + j*x + k*y];
        Pz[i + j*x + k*y] += d_dt * M[material_index].eta_tau_gamma_p * (Uxx[i + j*x + k*y] + Uyy[i + j*x + k*y] + Uzz[i + j*x + k*y]) - 2 * M[material_index].mu_tau_gamma_s * (Uxx[i + j*x + k*y] + Uyy[i + j*x + k*y]) + Rz[i + j*x + k*y] + F[i + j*x + k*y];
    }
}

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Pxy(float* Pxy, float* Ux, float* Uy, float* Rxy, unsigned int* S) {
    unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned int j = (blockIdx.y * blockDim.y) + threadIdx.y;
    unsigned int k = (blockIdx.z * blockDim.z) + threadIdx.z;

    if (i>=2 && i<x-2 && j>=2 && j<y-2 && k>=2 && k<z-2) {
        float Uxy = - Ux[i + (j+2)*x + k*y] + 27 * (Ux[i + (j+1)*x + k*y] - Ux[i + j*x + k*y]) + Ux[i + (j-1)*x + k*y];
        float Uyx = - Uy[i+2 + j*x + k*y] + 27 * (Uy[i+1 + j*x + k*y] - Uy[i + j*x + k*y]) + Uy[i-1 + j*x + k*y];
        unsigned int material_index = S[i + j*x + k*y];
        Pxy[i + j*x + k*y] += d_dt * M[material_index].mu_tau_gamma_s * (Uxy + Uyx) + Rxy[i + j*x + k*y];
    }
}

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Pyz(float* Pyz, float* Uy, float* Uz, float* Ryz, unsigned int* S) {
    unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned int j = (blockIdx.y * blockDim.y) + threadIdx.y;
    unsigned int k = (blockIdx.z * blockDim.z) + threadIdx.z;

    if (i>=2 && i<x-2 && j>=2 && j<y-2 && k>=2 && k<z-2) {
        float Uyz = - Uy[i + j*x + (k+2)*y] + 27 * (Uy[i + j*x + (k+1)*y] - Uy[i + j*x + k*y]) + Uy[i + j*x + (k-1)*y];
        float Uzy = - Uz[i + (j+2)*x + k*y] + 27 * (Uz[i + (j+1)*x + k*y] - Uz[i + j*x + k*y]) + Uz[i + (j-1)*x + k*y];
        unsigned int material_index = S[i + j*x + k*y];
        Pyz[i + j*x + k*y] += d_dt * M[material_index].mu_tau_gamma_s * (Uyz + Uzy) + Ryz[i + j*x + k*y];
    }
}

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Pxz(float* Pxz, float* Ux, float* Uz, float* Rxz, unsigned int* S) {
    unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned int j = (blockIdx.y * blockDim.y) + threadIdx.y;
    unsigned int k = (blockIdx.z * blockDim.z) + threadIdx.z;

    if (i>=2 && i<x-2 && j>=2 && j<y-2 && k>=2 && k<z-2) {
        float Uxz = - Ux[i + j*x + (k+2)*y] + 27 * (Ux[i + j*x + (k+1)*y] - Ux[i + j*x + k*y]) + Ux[i + j*x + (k-1)*y];
        float Uzx = - Uz[i+2 + j*x + k*y] + 27 * (Uz[i+1 + j*x + k*y] - Uz[i + j*x + k*y]) + Uz[i-1 + j*x + k*y];
        unsigned int material_index = S[i + j*x + k*y];
        Pxz[i + j*x + k*y] += d_dt * M[material_index].mu_tau_gamma_s * (Uxz + Uzx) + Rxz[i + j*x + k*y];
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