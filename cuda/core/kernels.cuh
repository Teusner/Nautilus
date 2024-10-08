#pragma once

#include <core/Material.cuh>
#include <core/Module.cuh>
#include <thrust/random.h>

#include <cuda.h>


template<unsigned int x, unsigned int y, unsigned int z, typename T>
__global__ void F(float t, T* E, float* F) {
    unsigned int index = threadIdx.x;
    T e = E[index];
    F[e.z + z*(e.y + y*e.x)] = e(t);
}

template <unsigned int x, unsigned int y, unsigned int z, unsigned int N>
struct Bound {
    float zeta_min, p;
    Bound (float _zeta_min, float _p) : zeta_min(_zeta_min), p(_p) {};

    __host__ __device__ float _value(int index, unsigned int size) const {
        return max(0.f, abs(index-size/2.f) - size/2.f + N) * M_PI / N;
    }

    __host__ __device__ float _f(float v) const {
        return (1 - zeta_min) * powf((1 + cosf(v)) / 2, p) + zeta_min;
    }

    __host__ __device__ float operator()(const int &index, float &v) const {
        int i = index / (y * z);
        int j = (index - z * y * i) / z;
        int k = (index - z * (y * i + j));

        return _f(_value(i, x)) * _f(_value(j, y)) * _f(_value(k, z));
    }
};

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Ux(float dt, float* alpha, float* Ux, float* Px, float* Pxy, float* Pxz, unsigned int* S, float* inv_rho) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i>=2 && i<x-2 && j>=2 && j<y-2 && k>=2 && k<z-2) {
        float dPxx = (- Px[k + z*(j + y*(i+2))] + 27 * (Px[k + z*(j + y*(i+1))] - Px[k + z*(j + y*i)]) + Px[k + z*(j + y*(i-1))]) * alpha[0];
        float dPxy = (- Pxy[k + z*(j+1 + y*i)] + 27 * (Pxy[k + z*(j + y*i)] - Pxy[k + z*(j-1 + y*i)]) + Pxy[k + z*(j-2 + y*i)]) * alpha[1];
        float dPxz = (- Pxz[k+1 + z*(j + y*i)] + 27 * (Pxz[k + z*(j + y*i)] - Pxz[k-1 + z*(j + y*i)]) + Pxz[k-2 + z*(j + y*i)]) * alpha[2];
        unsigned int material_index = S[k + z*(j + y*i)];
        Ux[k + z*(j + y*i)] += dt * inv_rho[material_index] * (dPxx + dPxy + dPxz);
    }
}

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Uy(float dt, float* alpha, float* Uy, float* Py, float* Pxy, float* Pyz, unsigned int* S, float* inv_rho) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i>=2 && i<x-2 && j>=2 && j<y-2 && k>=2 && k<z-2) {
        float dPyy = (- Py[k + z*(j+2 + y*i)] + 27 * (Py[k + z*(j+1 + y*i)] - Py[k + z*(j + y*i)]) + Py[k + z*(j-1 + y*i)]) * alpha[1];
        float dPxy = (- Pxy[k + z*(j + y*(i+1))] + 27 * (Pxy[k + z*(j + y*i)] - Pxy[k + z*(j + y*(i-1))]) + Pxy[k + z*(j + y*(i-2))]) * alpha[0];
        float dPyz = (- Pyz[k+1 + z*(j + y*i)] + 27 * (Pyz[k + z*(j + y*i)] - Pyz[k-1 + z*(j + y*i)]) + Pyz[k-2 + z*(j + y*i)]) * alpha[2];
        unsigned int material_index = S[k + z*(j + y*i)];
        Uy[k + z*(j + y*i)] += dt * inv_rho[material_index] * (dPyy + dPxy + dPyz);
    }
}

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Uz(float dt, float* alpha, float* Uz, float* Pz, float* Pyz, float* Pxz, unsigned int* S, float* inv_rho) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i>=2 && i<x-2 && j>=2 && j<y-2 && k>=2 && k<z-2) {
        float dPzz = (- Pz[k+2+ z*(j + y*i)] + 27 * (Pz[k+1 + z*(j + y*i)] - Pz[k + z*(j + y*i)]) + Pz[k-1 + z*(j + y*i)]) * alpha[2];
        float dPyz = (- Pyz[k + z*(j+1 + y*i)] + 27 * (Pyz[k + z*(j + y*i)] - Pyz[k + z*(j-1 + y*i)]) + Pyz[k+ z*(j-2 + y*i)]) * alpha[1];
        float dPxz = (- Pxz[k + z*(j + y*(i+1))] + 27 * (Pxz[k + z*(j + y*i)] - Pxz[k + z*(j + y*(i-1))]) + Pxz[k + z*(j + y*(i-2))]) * alpha[0];
        unsigned int material_index = S[k + z*(j + y*i)];
        Uz[k + z*(j + y*i)] += dt * inv_rho[material_index] * (dPzz + dPyz + dPxz);
    }
}

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Uxx(float alpha_x, float* Ux, float* Uxx) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i>=2 && i<x-2 && j>=2 && j<y-2 && k>=2 && k<z-2) {
        Uxx[k + z*(j + y*i)] = (- Ux[k + z*(j + y*(i+1))] + 27 * (Ux[k + z*(j + y*i)] - Ux[k + z*(j + y*(i-1))]) + Ux[k + z*(j + y*(i-2))]) * alpha_x;
    }
}

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Uyy(float alpha_y, float* Uy, float* Uyy) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i>=2 && i<x-2 && j>=2 && j<y-2 && k>=2 && k<z-2) {
        Uyy[k + z*(j + y*i)] = (- Uy[k + z*(j+1 + y*i)] + 27 * (Uy[k + z*(j + y*i)] - Uy[k + z*(j-1 + y*i)]) + Uy[k+ z*(j-2 + y*i)]) * alpha_y;
    }
}

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Uzz(float alpha_z, float* Uz, float* Uzz) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i>=2 && i<x-2 && j>=2 && j<y-2 && k>=2 && k<z-2) {
        Uzz[k + z*(j + y*i)] = (- Uz[k+1 + z*(j + y*i)] + 27 * (Uz[k + z*(j + y*i)] - Uz[k-1 + z*(j + y*i)]) + Uz[k-2 + z*(j + y*i)]) * alpha_z;
    }
}

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Rxx(float dt, float* Rx, float* Uxx, float* Uyy, float* Uzz, unsigned int* S, float* eta_tau_p, float* mu_tau_s, float* tau_sigma) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i>=2 && i<x-2 && j>=2 && j<y-2 && k>=2 && k<z-2) {
        unsigned int material_index = S[k + z*(j + y*i)];
        float a = (2 * tau_sigma[0] - dt) / (2 * tau_sigma[0] + dt);
        float b = (2 * dt) / (2 * tau_sigma[0] + dt);
        Rx[k + z*(j + y*i)] = a * Rx[k + z*(j + y*i)] - b * (eta_tau_p[material_index] * Uxx[k + z*(j + y*i)] + (eta_tau_p[material_index] - 2*mu_tau_s[material_index]) * (Uyy[k + z*(j + y*i)] + Uzz[k + z*(j + y*i)]));
    }
}

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Ryy(float dt, float* Ry, float* Uxx, float* Uyy, float* Uzz, unsigned int* S, float* eta_tau_p, float* mu_tau_s, float* tau_sigma) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i>=2 && i<x-2 && j>=2 && j<y-2 && k>=2 && k<z-2) {
        unsigned int material_index = S[k + z*(j + y*i)];
        float a = (2 * tau_sigma[0] - dt) / (2 * tau_sigma[0] + dt);
        float b = (2 * dt) / (2 * tau_sigma[0] + dt);
        Ry[k + z*(j + y*i)] = a * Ry[k + z*(j + y*i)] - b * (eta_tau_p[material_index] * Uyy[k + z*(j + y*i)] + (eta_tau_p[material_index] - 2*mu_tau_s[material_index]) * (Uxx[k + z*(j + y*i)] + Uzz[k + z*(j + y*i)]));
    }
}

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Rzz(float dt, float* Rz, float* Uxx, float* Uyy, float* Uzz, unsigned int* S, float* eta_tau_p, float* mu_tau_s, float* tau_sigma) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i>=2 && i<x-2 && j>=2 && j<y-2 && k>=2 && k<z-2) {
        unsigned int material_index = S[k + z*(j + y*i)];
        float a = (2 * tau_sigma[0] - dt) / (2 * tau_sigma[0] + dt);
        float b = (2 * dt) / (2 * tau_sigma[0] + dt);
        Rz[k + z*(j + y*i)] = a * Rz[k + z*(j + y*i)] - b * (eta_tau_p[material_index] * Uzz[k + z*(j + y*i)] + (eta_tau_p[material_index] - 2*mu_tau_s[material_index]) * (Uxx[k + z*(j + y*i)] + Uyy[k + z*(j + y*i)]));
    }
}

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Rxy(float dt, float* Rxy, float* Ux, float* Uy, unsigned int* S, float* mu_tau_s, float* tau_sigma) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i>=2 && i<x-2 && j>=2 && j<y-2 && k>=2 && k<z-2) {
        float Uxy = - Ux[k + z*(j+2 + y*i)] + 27 * (Ux[k + z*(j+1 + y*i)] - Ux[k + z*(j + y*i)]) + Ux[k + z*(j-1 + y*i)];
        float Uyx = - Uy[k+ z*(j + y*(i+2))] + 27 * (Uy[k + z*(j + y*(i+1))] - Uy[k + z*(j + y*i)]) + Uy[k + z*(j + y*(i-1))];
        unsigned int material_index = S[k + z*(j + y*i)];
        float a = (2 * tau_sigma[0] - dt) / (2 * tau_sigma[0] + dt);
        float b = (2 * dt) / (2 * tau_sigma[0] + dt);
        Rxy[k + z*(j + y*i)] = a * Rxy[k + z*(j + y*i)] - b * (mu_tau_s[material_index] * (Uxy + Uyx));
    }
}

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Ryz(float dt, float* Ryz, float* Uy, float* Uz, unsigned int* S, float* mu_tau_s, float* tau_sigma) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i>=2 && i<x-2 && j>=2 && j<y-2 && k>=2 && k<z-2) {
        float Uyz = - Uy[k+2+ z*(j + y*i)] + 27 * (Uy[k+1 + z*(j + y*i)] - Uy[k + z*(j + y*i)]) + Uy[k-1 + z*(j + y*i)];
        float Uzy = - Uz[k + z*(j+2 + y*i)] + 27 * (Uz[k + z*(j+1 + y*i)] - Uz[k + z*(j + y*i)]) + Uz[k + z*(j-1 + y*i)];
        unsigned int material_index = S[k + z*(j + y*i)];
        float a = (2 * tau_sigma[0] - dt) / (2 * tau_sigma[0] + dt);
        float b = (2 * dt) / (2 * tau_sigma[0] + dt);
        Ryz[k + z*(j + y*i)] = a * Ryz[k + z*(j + y*i)] - b * (mu_tau_s[material_index] * (Uyz + Uzy));
    }
}

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Rxz(float dt, float* Rxz, float* Ux, float* Uz, unsigned int* S, float* mu_tau_s, float* tau_sigma) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i>=2 && i<x-2 && j>=2 && j<y-2 && k>=2 && k<z-2) {
        float Uxz = - Ux[k+2+ z*(j + y*i)] + 27 * (Ux[k+1 + z*(j + y*i)] - Ux[k + z*(j + y*i)]) + Ux[k-1 + z*(j + y*i)];
        float Uzx = - Uz[k+ z*(j + y*(i+2))] + 27 * (Uz[k + z*(j + y*(i+1))] - Uz[k + z*(j + y*i)]) + Uz[k + z*(j + y*(i-1))];
        unsigned int material_index = S[k + z*(j + y*i)];
        float a = (2 * tau_sigma[0] - dt) / (2 * tau_sigma[0] + dt);
        float b = (2 * dt) / (2 * tau_sigma[0] + dt);
        Rxz[k + z*(j + y*i)] = a * Rxz[k + z*(j + y*i)] - b * (mu_tau_s[material_index] * (Uxz + Uzx));
    }
}

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Pxx(float dt, float* Px, float* Uxx, float* Uyy, float* Uzz, float* Rx, unsigned int* S, float* eta_tau_p, float* mu_tau_s, float* F) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i>=2 && i<x-2 && j>=2 && j<y-2 && k>=2 && k<z-2) {
        unsigned int material_index = S[k + z*(j + y*i)];
        Px[k + z*(j + y*i)] += dt * (eta_tau_p[material_index] * Uxx[k + z*(j + y*i)] + (eta_tau_p[material_index] - 2*mu_tau_s[material_index]) * (Uyy[k + z*(j + y*i)] + Uzz[k + z*(j + y*i)]) - 0.5 * Rx[k + z*(j + y*i)] - F[k + z*(j + y*i)]);
    }
}

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Pyy(float dt, float* Py, float* Uxx, float* Uyy, float* Uzz, float* Ry, unsigned int* S, float* eta_tau_p, float* mu_tau_s, float* F) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i>=2 && i<x-2 && j>=2 && j<y-2 && k>=2 && k<z-2) {
        unsigned int material_index = S[k + z*(j + y*i)];
        Py[k + z*(j + y*i)] += dt * (eta_tau_p[material_index] * Uyy[k + z*(j + y*i)] + (eta_tau_p[material_index] - 2*mu_tau_s[material_index]) * (Uxx[k + z*(j + y*i)] + Uzz[k + z*(j + y*i)]) - 0.5 * Ry[k + z*(j + y*i)] - F[k + z*(j + y*i)]);
    }
}

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Pzz(float dt, float* Pz, float* Uxx, float* Uyy, float* Uzz, float* Rz, unsigned int* S, float* eta_tau_p, float* mu_tau_s, float* F) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i>=2 && i<x-2 && j>=2 && j<y-2 && k>=2 && k<z-2) {
        unsigned int material_index = S[k + z*(j + y*i)];
        Pz[k + z*(j + y*i)] += dt * (eta_tau_p[material_index] * Uzz[k + z*(j + y*i)] + (eta_tau_p[material_index] - 2*mu_tau_s[material_index]) * (Uxx[k + z*(j + y*i)] + Uyy[k + z*(j + y*i)]) - 0.5 * Rz[k + z*(j + y*i)] - F[k + z*(j + y*i)]);
    }
}

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Pxy(float dt, float* Pxy, float* Ux, float* Uy, float* Rxy, unsigned int* S, float* mu_tau_s) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i>=2 && i<x-2 && j>=2 && j<y-2 && k>=2 && k<z-2) {
        float Uxy = - Ux[k + z*(j+2 + y*i)] + 27 * (Ux[k + z*(j+1 + y*i)] - Ux[k + z*(j + y*i)]) + Ux[k + z*(j-1 + y*i)];
        float Uyx = - Uy[k+ z*(j + y*(i+2))] + 27 * (Uy[k + z*(j + y*(i+1))] - Uy[k + z*(j + y*i)]) + Uy[k + z*(j + y*(i-1))];
        unsigned int material_index = S[k + z*(j + y*i)];
        Pxy[k + z*(j + y*i)] += dt * (mu_tau_s[material_index] * (Uxy + Uyx) - 0.5 * Rxy[k + z*(j + y*i)]);
    }
}

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Pyz(float dt, float* Pyz, float* Uy, float* Uz, float* Ryz, unsigned int* S, float* mu_tau_s) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i>=2 && i<x-2 && j>=2 && j<y-2 && k>=2 && k<z-2) {
        float Uyz = - Uy[k+2+ z*(j + y*i)] + 27 * (Uy[k+1 + z*(j + y*i)] - Uy[k + z*(j + y*i)]) + Uy[k-1 + z*(j + y*i)];
        float Uzy = - Uz[k + z*(j+2 + y*i)] + 27 * (Uz[k + z*(j+1 + y*i)] - Uz[k + z*(j + y*i)]) + Uz[k + z*(j-1 + y*i)];
        unsigned int material_index = S[k + z*(j + y*i)];
        Pyz[k + z*(j + y*i)] += dt * (mu_tau_s[material_index] * (Uyz + Uzy) - 0.5 * Ryz[k + z*(j + y*i)]);
    }
}

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Pxz(float dt, float* Pxz, float* Ux, float* Uz, float* Rxz, unsigned int* S, float* mu_tau_s) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i>=2 && i<x-2 && j>=2 && j<y-2 && k>=2 && k<z-2) {
        float Uxz = - Ux[k+2+ z*(j + y*i)] + 27 * (Ux[k+1 + z*(j + y*i)] - Ux[k + z*(j + y*i)]) + Ux[k-1 + z*(j + y*i)];
        float Uzx = - Uz[k+ z*(j + y*(i+2))] + 27 * (Uz[k + z*(j + y*(i+1))] - Uz[k + z*(j + y*i)]) + Uz[k + z*(j + y*(i-1))];
        unsigned int material_index = S[k + z*(j + y*i)];
        Pxz[k + z*(j + y*i)] += dt * (mu_tau_s[material_index] * (Uxz + Uzx) - 0.5 * Rxz[k + z*(j + y*i)]);
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

struct saxpy_functor {
    const float a;
    saxpy_functor(float _a) : a(_a) {}
    __host__ __device__ float operator()(const float& x, const float& y) const { return a * x + y; }
};