#pragma once

#include <core/Material.cuh>
#include <core/Module.cuh>
#include <thrust/random.h>


template<unsigned int x, unsigned int y, unsigned int z>
__global__ void F(float t, Emitter* E, float* F);

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Ux(float dt, float* Ux, float* Px, float* Pxy, float* Pxz, unsigned int* S, float* inv_rho, float* alpha);

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Uy(float dt, float* Uy, float* Py, float* Pxy, float* Pyz, unsigned int* S, float* inv_rho, float* alpha);

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Uz(float dt, float* Uz, float* Pz, float* Pyz, float* Pxz, unsigned int* S, float* inv_rho, float* alpha);

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Uxx(float* Ux, float* Uxx, float alpha_x);

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Uyy(float* Uy, float* Uyy, float alpha_y);

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Uzz(float* Uz, float* Uzz, float alpha_z);

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Rxx(float dt, float* Rx, float* Ux, float* Uy, float* Uz, unsigned int* S, float* eta_tau_gamma_p, float* mu_tau_gamma_s, float* tau_sigma);

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Ryy(float dt, float* Ry, float* Ux, float* Uy, float* Uz, unsigned int* S, float* eta_tau_gamma_p, float* mu_tau_gamma_s, float* tau_sigma);

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Rzz(float dt, float* Rz, float* Ux, float* Uy, float* Uz, unsigned int* S, float* eta_tau_gamma_p, float* mu_tau_gamma_s, float* tau_sigma);

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Rxy(float dt, float* Rxy, float* Ux, float* Uy, unsigned int* S, float* mu_tau_gamma_s, float* tau_sigma);

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Ryz(float dt, float* Ryz, float* Uy, float* Uz, unsigned int* S, float* mu_tau_gamma_s, float* tau_sigma);

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Rxz(float dt, float* Rxz, float* Ux, float* Uz, unsigned int* S, float* mu_tau_gamma_s, float* tau_sigma);

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Pxx(float dt, float* Px, float* Ux, float* Uy, float* Uz, float* Rx, unsigned int* S, float* eta_tau_gamma_p, float* mu_tau_gamma_s, float* F);

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Pyy(float dt, float* Py, float* Ux, float* Uy, float* Uz, float* Rx, unsigned int* S, float* eta_tau_gamma_p, float* mu_tau_gamma_s, float* F);

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Pzz(float dt, float* Pz, float* Ux, float* Uy, float* Uz, float* Rz, unsigned int* S, float* eta_tau_gamma_p, float* mu_tau_gamma_s, float* F);

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Pxy(float dt, float* Pxy, float* Ux, float* Uy, float* Rxy, unsigned int* S, float* mu_tau_gamma_s);

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Pyz(float dt, float* Pyz, float* Uy, float* Uz, float* Ryz, unsigned int* S, float* mu_tau_gamma_s);

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Pxz(float dt, float* Pxz, float* Ux, float* Uz, float* Rxz, unsigned int* S, float* mu_tau_gamma_s);


/// Implementation
template<unsigned int x, unsigned int y, unsigned int z, typename T>
__global__ void F(float t, T* E, float* F) {
    unsigned int index = threadIdx.x;
    T e = E[index];
    // printf("%f\n", e(t));
    F[e.x + e.y * x + e.z * y] = e(t);
}

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Ux(float dt, float* Ux, float* Px, float* Pxy, float* Pxz, unsigned int* S, float* inv_rho, float* alpha) {
    unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned int j = (blockIdx.y * blockDim.y) + threadIdx.y;
    unsigned int k = (blockIdx.z * blockDim.z) + threadIdx.z;

    if (i>=2 && i<x-2 && j>=2 && j<y-2 && k>=2 && k<z-2) {
        float dPxx = (- Px[i+2 + j*x + k*y] + 27 * (Px[i+1 + j*x + k*y] - Px[i + j*x + k*y]) + Px[i-1 + j*x + k*y]) * alpha[0];
        float dPxy = (- Pxy[i + (j+1)*x + k*y] + 27 * (Pxy[i + j*x + k*y] - Pxy[i + (j-1)*x + k*y]) + Pxy[i + (j-2)*x + k*y]) * alpha[1];
        float dPxz = (- Pxz[i + j*x + (k+1)*y] + 27 * (Pxz[i + j*x + k*y] - Pxz[i + j*x + (k-1)*y]) + Pxz[i + j*x + (k-2)*y]) * alpha[2];
        unsigned int material_index = S[i + j*x + k*y];
        Ux[i + j*x + k*y] += dt * inv_rho[material_index] * (dPxx + dPxy + dPxz);
    }
}

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Uy(float dt, float* Uy, float* Py, float* Pxy, float* Pyz, unsigned int* S, float* inv_rho, float* alpha) {
    unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned int j = (blockIdx.y * blockDim.y) + threadIdx.y;
    unsigned int k = (blockIdx.z * blockDim.z) + threadIdx.z;

    if (i>=2 && i<x-2 && j>=2 && j<y-2 && k>=2 && k<z-2) {
        float dPyy = (- Py[i + (j+2)*x + k*y] + 27 * (Py[i + (j+1)*x + k*y] - Py[i + j*x + k*y]) + Py[i + (j-1)*x + k*y]) * alpha[1];
        float dPxy = (- Pxy[i+1 + j*x + k*y] + 27 * (Pxy[i + j*x + k*y] - Pxy[i-1 + j*x + k*y]) + Pxy[i-2 + j*x + k*y]) * alpha[0];
        float dPyz = (- Pyz[i + j*x + (k+1)*y] + 27 * (Pyz[i + j*x + k*y] - Pyz[i + j*x + (k-1)*y]) + Pyz[i + j*x + (k-2)*y]) * alpha[2];
        unsigned int material_index = S[i + j*x + k*y];
        Uy[i + j*x + k*y] += dt * inv_rho[material_index] * (dPyy + dPxy + dPyz);
    }
}

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Uz(float dt, float* Uz, float* Pz, float* Pyz, float* Pxz, unsigned int* S, float* inv_rho, float* alpha) {
    unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned int j = (blockIdx.y * blockDim.y) + threadIdx.y;
    unsigned int k = (blockIdx.z * blockDim.z) + threadIdx.z;

    if (i>=2 && i<x-2 && j>=2 && j<y-2 && k>=2 && k<z-2) {
        float dPzz = (- Pz[i + j*x + (k+2)*y] + 27 * (Pz[i + j*x + (k+1)*y] - Pz[i + j*x + k*y]) + Pz[i + j*x + (k-1)*y]) * alpha[2];
        float dPyz = (- Pyz[i+1 + j*x + k*y] + 27 * (Pyz[i + j*x + k*y] - Pyz[i-1 + j*x + k*y]) + Pyz[i-2 + j*x + k*y]) * alpha[0];
        float dPxz = (- Pxz[i + (j+1)*x + k*y] + 27 * (Pxz[i + j*x + k*y] - Pxz[i + (j-1)*x + k*y]) + Pxz[i + (j-2)*x + k*y]) * alpha[1];
        unsigned int material_index = S[i + j*x + k*y];
        Uz[i + j*x + k*y] += dt * inv_rho[material_index] * (dPzz + dPyz + dPxz);
    }
}

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Uxx(float alpha_x, float* Ux, float* Uxx) {
    unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned int j = (blockIdx.y * blockDim.y) + threadIdx.y;
    unsigned int k = (blockIdx.z * blockDim.z) + threadIdx.z;

    if (i>=2 && i<x-2 && j>=2 && j<y-2 && k>=2 && k<z-2) {
        Uxx[i + j*x + k*y] = (- Ux[i+1 + j*x + k*y] + 27 * (Ux[i + j*x + k*y] - Ux[i-1 + j*x + k*y]) + Ux[i-2 + j*x + k*y]) * alpha_x;
    }
}

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Uyy(float alpha_y, float* Uy, float* Uyy) {
    unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned int j = (blockIdx.y * blockDim.y) + threadIdx.y;
    unsigned int k = (blockIdx.z * blockDim.z) + threadIdx.z;

    if (i>=2 && i<x-2 && j>=2 && j<y-2 && k>=2 && k<z-2) {
        Uyy[i + j*x + k*y] = (- Uy[i + (j+1)*x + k*y] + 27 * (Uy[i + j*x + k*y] - Uy[i + (j-1)*x + k*y]) + Uy[i + (j-2)*x + k*y]) * alpha_y;
    }
}

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Uzz(float alpha_z, float* Uz, float* Uzz) {
    unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned int j = (blockIdx.y * blockDim.y) + threadIdx.y;
    unsigned int k = (blockIdx.z * blockDim.z) + threadIdx.z;

    if (i>=2 && i<x-2 && j>=2 && j<y-2 && k>=2 && k<z-2) {
        Uzz[i + j*x + k*y] = (- Uz[i + j*x + (k+1)*y] + 27 * (Uz[i + j*x + k*y] - Uz[i + j*x + (k-1)*y]) + Uz[i + j*x + (k-2)*y]) * alpha_z;
    }
}

template<unsigned int x, unsigned int y, unsigned int z, unsigned int l>
__global__ void Rxx(float dt, float* Rx, float* Uxx, float* Uyy, float* Uzz, unsigned int* S, float* eta_tau_gamma_p, float* mu_tau_gamma_s, float* tau_sigma) {
    unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned int j = (blockIdx.y * blockDim.y) + threadIdx.y;
    unsigned int k = (blockIdx.z * blockDim.z) + threadIdx.z;

    if (i>=2 && i<x-2 && j>=2 && j<y-2 && k>=2 && k<z-2) {
        unsigned int material_index = S[i + j*x + k*y];
        #pragma unroll
        for (unsigned int q=0; q < l; ++q) {
            float a = (2 * tau_sigma[q] - dt) / (2 * tau_sigma[q] + dt);
            float b = 2 * dt / (2 * tau_sigma[q] + dt);
            Rx[i + j*x + k*y + q*z] += a * Rx[i + j*x + k*y + q*z] - b * eta_tau_gamma_p[material_index] * (Uxx[i + j*x + k*y] + Uyy[i + j*x + k*y] + Uzz[i + j*x + k*y]) - 2 * mu_tau_gamma_s[material_index] * (Uyy[i + j*x + k*y] + Uzz[i + j*x + k*y]);
        }
    }
}

template<unsigned int x, unsigned int y, unsigned int z, unsigned int l>
__global__ void Ryy(float dt, float* Ry, float* Uxx, float* Uyy, float* Uzz, unsigned int* S, float* eta_tau_gamma_p, float* mu_tau_gamma_s, float* tau_sigma) {
    unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned int j = (blockIdx.y * blockDim.y) + threadIdx.y;
    unsigned int k = (blockIdx.z * blockDim.z) + threadIdx.z;

    if (i>=2 && i<x-2 && j>=2 && j<y-2 && k>=2 && k<z-2) {
        unsigned int material_index = S[i + j*x + k*y];
        #pragma unroll
        for (unsigned int q=0; q < l; ++q) {
            float a = (2 * tau_sigma[q] - dt) / (2 * tau_sigma[q] + dt);
            float b = 2 * dt / (2 * tau_sigma[q] + dt);
            Ry[i + j*x + k*y + q*z] += a * Ry[i + j*x + k*y + q*z] - b * eta_tau_gamma_p[material_index] * (Uxx[i + j*x + k*y] + Uyy[i + j*x + k*y] + Uzz[i + j*x + k*y]) - 2 * mu_tau_gamma_s[material_index] * (Uxx[i + j*x + k*y] + Uzz[i + j*x + k*y]);
        }
    }
}

template<unsigned int x, unsigned int y, unsigned int z, unsigned int l>
__global__ void Rzz(float dt, float* Rz, float* Uxx, float* Uyy, float* Uzz, unsigned int* S, float* eta_tau_gamma_p, float* mu_tau_gamma_s, float* tau_sigma) {
    unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned int j = (blockIdx.y * blockDim.y) + threadIdx.y;
    unsigned int k = (blockIdx.z * blockDim.z) + threadIdx.z;

    if (i>=2 && i<x-2 && j>=2 && j<y-2 && k>=2 && k<z-2) {
        unsigned int material_index = S[i + j*x + k*y];
        #pragma unroll
        for (unsigned int q=0; q < l; ++q) {
            float a = (2 * tau_sigma[q] - dt) / (2 * tau_sigma[q] + dt);
            float b = 2 * dt / (2 * tau_sigma[q] + dt);
            Rz[i + j*x + k*y + q*z] += a * Rz[i + j*x + k*y + q*z] - b * eta_tau_gamma_p[material_index] * (Uxx[i + j*x + k*y] + Uyy[i + j*x + k*y] + Uzz[i + j*x + k*y]) - 2 * mu_tau_gamma_s[material_index] * (Uxx[i + j*x + k*y] + Uyy[i + j*x + k*y]);
        }
    }
}

template<unsigned int x, unsigned int y, unsigned int z, unsigned int l>
__global__ void Rxy(float dt, float* Rxy, float* Ux, float* Uy, unsigned int* S, float* mu_tau_gamma_s, float* tau_sigma) {
    unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned int j = (blockIdx.y * blockDim.y) + threadIdx.y;
    unsigned int k = (blockIdx.z * blockDim.z) + threadIdx.z;

    if (i>=2 && i<x-2 && j>=2 && j<y-2 && k>=2 && k<z-2) {
        float Uxy = - Ux[i + (j+2)*x + k*y] + 27 * (Ux[i + (j+1)*x + k*y] - Ux[i + j*x + k*y]) + Ux[i + (j-1)*x + k*y];
        float Uyx = - Uy[i+2 + j*x + k*y] + 27 * (Uy[i+1 + j*x + k*y] - Uy[i + j*x + k*y]) + Uy[i-1 + j*x + k*y];
        unsigned int material_index = S[i + j*x + k*y];
        #pragma unroll
        for (unsigned int q=0; q < l; ++q) {
            float a = (2 * tau_sigma[q] - dt) / (2 * tau_sigma[q] + dt);
            float b = 2 * dt / (2 * tau_sigma[q] + dt);
            Rxy[i + j*x + k*y + q*z] += a * Rxy[i + j*x + k*y + q*z] - b * mu_tau_gamma_s[material_index] * (Uxy + Uyx);
        }
    }
}

template<unsigned int x, unsigned int y, unsigned int z, unsigned int l>
__global__ void Ryz(float dt, float* Ryz, float* Uy, float* Uz, unsigned int* S, float* mu_tau_gamma_s, float* tau_sigma) {
    unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned int j = (blockIdx.y * blockDim.y) + threadIdx.y;
    unsigned int k = (blockIdx.z * blockDim.z) + threadIdx.z;

    if (i>=2 && i<x-2 && j>=2 && j<y-2 && k>=2 && k<z-2) {
        float Uyz = - Uy[i + j*x + (k+2)*y] + 27 * (Uy[i + j*x + (k+1)*y] - Uy[i + j*x + k*y]) + Uy[i + j*x + (k-1)*y];
        float Uzy = - Uz[i + (j+2)*x + k*y] + 27 * (Uz[i + (j+1)*x + k*y] - Uz[i + j*x + k*y]) + Uz[i + (j-1)*x + k*y];
        unsigned int material_index = S[i + j*x + k*y];
        #pragma unroll
        for (unsigned int q=0; q < l; ++q) {
            float a = (2 * tau_sigma[q] - dt) / (2 * tau_sigma[q] + dt);
            float b = 2 * dt / (2 * tau_sigma[q] + dt);
            Ryz[i + j*x + k*y + q*z] += a * Ryz[i + j*x + k*y + q*z] - b * mu_tau_gamma_s[material_index] * (Uyz + Uzy);
        }
    }
}

template<unsigned int x, unsigned int y, unsigned int z, unsigned int l>
__global__ void Rxz(float dt, float* Rxz, float* Ux, float* Uz, unsigned int* S, float* mu_tau_gamma_s, float* tau_sigma) {
    unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned int j = (blockIdx.y * blockDim.y) + threadIdx.y;
    unsigned int k = (blockIdx.z * blockDim.z) + threadIdx.z;

    if (i>=2 && i<x-2 && j>=2 && j<y-2 && k>=2 && k<z-2) {
        float Uxz = - Ux[i + j*x + (k+2)*y] + 27 * (Ux[i + j*x + (k+1)*y] - Ux[i + j*x + k*y]) + Ux[i + j*x + (k-1)*y];
        float Uzx = - Uz[i+2 + j*x + k*y] + 27 * (Uz[i+1 + j*x + k*y] - Uz[i + j*x + k*y]) + Uz[i-1 + j*x + k*y];
        unsigned int material_index = S[i + j*x + k*y];
        #pragma unroll
        for (unsigned int q=0; q < l; ++q) {
            float a = (2 * tau_sigma[q] - dt) / (2 * tau_sigma[q] + dt);
            float b = 2 * dt / (2 * tau_sigma[q] + dt);
            Rxz[i + j*x + k*y + q*z] += a * Rxz[i + j*x + k*y + q*z] - b * mu_tau_gamma_s[material_index] * (Uxz + Uzx);
        }
    }
}

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Pxx(float dt, float* Px, float* Uxx, float* Uyy, float* Uzz, float* Rx, unsigned int* S, float* eta_tau_gamma_p, float* mu_tau_gamma_s, float* F) {
    unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned int j = (blockIdx.y * blockDim.y) + threadIdx.y;
    unsigned int k = (blockIdx.z * blockDim.z) + threadIdx.z;

    if (i>=2 && i<x-2 && j>=2 && j<y-2 && k>=2 && k<z-2) {
        unsigned int material_index = S[i + j*x + k*y];
        Px[i + j*x + k*y] += dt * (eta_tau_gamma_p[material_index] * (Uxx[i + j*x + k*y] + Uyy[i + j*x + k*y] + Uzz[i + j*x + k*y]) - 2 * mu_tau_gamma_s[material_index] * (Uyy[i + j*x + k*y] + Uzz[i + j*x + k*y]) + Rx[i + j*x + k*y]) + F[i + j*x + k*y];
    }
}

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Pyy(float dt, float* Py, float* Uxx, float* Uyy, float* Uzz, float* Ry, unsigned int* S, float* eta_tau_gamma_p, float* mu_tau_gamma_s, float* F) {
    unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned int j = (blockIdx.y * blockDim.y) + threadIdx.y;
    unsigned int k = (blockIdx.z * blockDim.z) + threadIdx.z;

    if (i>=2 && i<x-2 && j>=2 && j<y-2 && k>=2 && k<z-2) {
        unsigned int material_index = S[i + j*x + k*y];
        Py[i + j*x + k*y] += dt * (eta_tau_gamma_p[material_index] * (Uxx[i + j*x + k*y] + Uyy[i + j*x + k*y] + Uzz[i + j*x + k*y]) - 2 * mu_tau_gamma_s[material_index] * (Uxx[i + j*x + k*y] + Uzz[i + j*x + k*y]) + Ry[i + j*x + k*y]) + F[i + j*x + k*y];
    }
}

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Pzz(float dt, float* Pz, float* Uxx, float* Uyy, float* Uzz, float* Rz, unsigned int* S, float* eta_tau_gamma_p, float* mu_tau_gamma_s, float* F) {
    unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned int j = (blockIdx.y * blockDim.y) + threadIdx.y;
    unsigned int k = (blockIdx.z * blockDim.z) + threadIdx.z;

    if (i>=2 && i<x-2 && j>=2 && j<y-2 && k>=2 && k<z-2) {
        unsigned int material_index = S[i + j*x + k*y];
        Pz[i + j*x + k*y] += dt * (eta_tau_gamma_p[material_index] * (Uxx[i + j*x + k*y] + Uyy[i + j*x + k*y] + Uzz[i + j*x + k*y]) - 2 * mu_tau_gamma_s[material_index] * (Uxx[i + j*x + k*y] + Uyy[i + j*x + k*y]) + Rz[i + j*x + k*y]) + F[i + j*x + k*y];
    }
}

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Pxy(float dt, float* Pxy, float* Ux, float* Uy, float* Rxy, unsigned int* S, float* mu_tau_gamma_s) {
    unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned int j = (blockIdx.y * blockDim.y) + threadIdx.y;
    unsigned int k = (blockIdx.z * blockDim.z) + threadIdx.z;

    if (i>=2 && i<x-2 && j>=2 && j<y-2 && k>=2 && k<z-2) {
        float Uxy = - Ux[i + (j+2)*x + k*y] + 27 * (Ux[i + (j+1)*x + k*y] - Ux[i + j*x + k*y]) + Ux[i + (j-1)*x + k*y];
        float Uyx = - Uy[i+2 + j*x + k*y] + 27 * (Uy[i+1 + j*x + k*y] - Uy[i + j*x + k*y]) + Uy[i-1 + j*x + k*y];
        unsigned int material_index = S[i + j*x + k*y];
        Pxy[i + j*x + k*y] += dt * (mu_tau_gamma_s[material_index] * (Uxy + Uyx) + Rxy[i + j*x + k*y]);
    }
}

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Pyz(float dt, float* Pyz, float* Uy, float* Uz, float* Ryz, unsigned int* S, float* mu_tau_gamma_s) {
    unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned int j = (blockIdx.y * blockDim.y) + threadIdx.y;
    unsigned int k = (blockIdx.z * blockDim.z) + threadIdx.z;

    if (i>=2 && i<x-2 && j>=2 && j<y-2 && k>=2 && k<z-2) {
        float Uyz = - Uy[i + j*x + (k+2)*y] + 27 * (Uy[i + j*x + (k+1)*y] - Uy[i + j*x + k*y]) + Uy[i + j*x + (k-1)*y];
        float Uzy = - Uz[i + (j+2)*x + k*y] + 27 * (Uz[i + (j+1)*x + k*y] - Uz[i + j*x + k*y]) + Uz[i + (j-1)*x + k*y];
        unsigned int material_index = S[i + j*x + k*y];
        Pyz[i + j*x + k*y] += dt * (mu_tau_gamma_s[material_index] * (Uyz + Uzy) + Ryz[i + j*x + k*y]);
    }
}

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Pxz(float dt, float* Pxz, float* Ux, float* Uz, float* Rxz, unsigned int* S, float* mu_tau_gamma_s) {
    unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned int j = (blockIdx.y * blockDim.y) + threadIdx.y;
    unsigned int k = (blockIdx.z * blockDim.z) + threadIdx.z;

    if (i>=2 && i<x-2 && j>=2 && j<y-2 && k>=2 && k<z-2) {
        float Uxz = - Ux[i + j*x + (k+2)*y] + 27 * (Ux[i + j*x + (k+1)*y] - Ux[i + j*x + k*y]) + Ux[i + j*x + (k-1)*y];
        float Uzx = - Uz[i+2 + j*x + k*y] + 27 * (Uz[i+1 + j*x + k*y] - Uz[i + j*x + k*y]) + Uz[i-1 + j*x + k*y];
        unsigned int material_index = S[i + j*x + k*y];
        Pxz[i + j*x + k*y] += dt * (mu_tau_gamma_s[material_index] * (Uxz + Uzx) + Rxz[i + j*x + k*y]);
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