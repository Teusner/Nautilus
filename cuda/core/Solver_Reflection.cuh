#pragma once
#include "Scene.cuh"
#include "kernels.cuh"
#include <thrust/device_vector.h>
#include <thrust/device_malloc.h>

template<unsigned int x, unsigned int y, unsigned int z, unsigned int N, typename T, typename S>
class Scene;

template <unsigned int x, unsigned int y, unsigned int z, unsigned int N>
struct RBound {
    float zeta_min, p;
    RBound (float _zeta_min, float _p) : zeta_min(_zeta_min), p(_p) {};

    __host__ __device__ float _value(int index, unsigned int size) const {
        return max(0.f, abs(index-size/2.f) - size/2.f + N) * M_PI / N;
    }

    __host__ __device__ float _rvalue(int index, unsigned int size) const {
        return max(0.f, float(index) - float(size) + float(N)) * M_PI / float(N);
    }

    __host__ __device__ float _f(float v) const {
        return (1 - zeta_min) * powf((1 + cosf(v)) / 2, p) + zeta_min;
    }

    __host__ __device__ float operator()(const int &index, float &v) const {
        int i = index / (y * z);
        int j = (index - z * y * i) / z;
        int k = (index - z * (y * i + j));

        return _f(_rvalue(i, x)) * _f(_rvalue(j, y)) * _f(_value(k, z));
    }
};


template<unsigned int x, unsigned int y, unsigned int z>
__global__ void RUx(float dt, float* alpha, float* Ux, float* Px, float* Pxy, float* Pxz, unsigned int* S, float* inv_rho) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i>=0 && i<x-2 && j>=0 && j<y-2 && k>=2 && k<z-2) {
        float dPxx = (- Px[k + z*(j + y*(i+2))] + 27 * (Px[k + z*(j + y*(i+1))] - Px[k + z*(j + y*i)]) + Px[k + z*(j + y*abs(int(i)-1))]) * alpha[0];
        float dPxy = (- Pxy[k + z*(j+1 + y*i)] + 27 * (Pxy[k + z*(j + y*i)] - Pxy[k + z*(abs(int(j)-1) + y*i)]) + Pxy[k + z*(abs(int(j)-2) + y*i)]) * alpha[1];
        float dPxz = (- Pxz[k+1 + z*(j + y*i)] + 27 * (Pxz[k + z*(j + y*i)] - Pxz[k-1 + z*(j + y*i)]) + Pxz[k-2 + z*(j + y*i)]) * alpha[2];
        unsigned int material_index = S[k + z*(j + y*i)];
        Ux[k + z*(j + y*i)] += dt * inv_rho[material_index] * (dPxx + dPxy + dPxz);
    }
}

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void RUy(float dt, float* alpha, float* Uy, float* Py, float* Pxy, float* Pyz, unsigned int* S, float* inv_rho) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i>=0 && i<x-2 && j>=0 && j<y-2 && k>=2 && k<z-2) {
        float dPyy = (- Py[k + z*(j+2 + y*i)] + 27 * (Py[k + z*(j+1 + y*i)] - Py[k + z*(j + y*i)]) + Py[k + z*(abs(int(j)-1) + y*i)]) * alpha[1];
        float dPxy = (- Pxy[k + z*(j + y*(i+1))] + 27 * (Pxy[k + z*(j + y*i)] - Pxy[k + z*(j + y*abs(int(i)-1))]) + Pxy[k + z*(j + y*abs(int(i)-2))]) * alpha[0];
        float dPyz = (- Pyz[k+1 + z*(j + y*i)] + 27 * (Pyz[k + z*(j + y*i)] - Pyz[k-1 + z*(j + y*i)]) + Pyz[k-2 + z*(j + y*i)]) * alpha[2];
        unsigned int material_index = S[k + z*(j + y*i)];
        Uy[k + z*(j + y*i)] += dt * inv_rho[material_index] * (dPyy + dPxy + dPyz);
    }
}

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void RUz(float dt, float* alpha, float* Uz, float* Pz, float* Pyz, float* Pxz, unsigned int* S, float* inv_rho) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i>=0 && i<x-2 && j>=0 && j<y-2 && k>=2 && k<z-2) {
        float dPzz = (- Pz[k+2+ z*(j + y*i)] + 27 * (Pz[k+1 + z*(j + y*i)] - Pz[k + z*(j + y*i)]) + Pz[k-1 + z*(j + y*i)]) * alpha[2];
        float dPyz = (- Pyz[k + z*(j+1 + y*i)] + 27 * (Pyz[k + z*(j + y*i)] - Pyz[k + z*(abs(int(j)-1) + y*i)]) + Pyz[k+ z*(abs(int(j)-2) + y*i)]) * alpha[1];
        float dPxz = (- Pxz[k + z*(j + y*(i+1))] + 27 * (Pxz[k + z*(j + y*i)] - Pxz[k + z*(j + y*abs(int(i)-1))]) + Pxz[k + z*(j + y*abs(int(i)-2))]) * alpha[0];
        unsigned int material_index = S[k + z*(j + y*i)];
        Uz[k + z*(j + y*i)] += dt * inv_rho[material_index] * (dPzz + dPyz + dPxz);
    }
}

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void RUxx(float alpha_x, float* Ux, float* Uxx) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i>=0 && i<x-2 && j>=0 && j<y-2 && k>=2 && k<z-2) {
        Uxx[k + z*(j + y*i)] = (- Ux[k + z*(j + y*(i+1))] + 27 * (Ux[k + z*(j + y*i)] - Ux[k + z*(j + y*abs(int(i)-1))]) + Ux[k + z*(j + y*abs(int(i)-2))]) * alpha_x;
    }
}

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void RUyy(float alpha_y, float* Uy, float* Uyy) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i>=0 && i<x-2 && j>=0 && j<y-2 && k>=2 && k<z-2) {
        Uyy[k + z*(j + y*i)] = (- Uy[k + z*(j+1 + y*i)] + 27 * (Uy[k + z*(j + y*i)] - Uy[k + z*(abs(int(j)-1) + y*i)]) + Uy[k+ z*(abs(int(j)-2) + y*i)]) * alpha_y;
    }
}

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void RUzz(float alpha_z, float* Uz, float* Uzz) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i>=0 && i<x-2 && j>=0 && j<y-2 && k>=2 && k<z-2) {
        Uzz[k + z*(j + y*i)] = (- Uz[k+1 + z*(j + y*i)] + 27 * (Uz[k + z*(j + y*i)] - Uz[k-1 + z*(j + y*i)]) + Uz[k-2 + z*(j + y*i)]) * alpha_z;
    }
}

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void RRxx(float dt, float* Rx, float* Uxx, float* Uyy, float* Uzz, unsigned int* S, float* eta_tau_p, float* mu_tau_s, float* tau_sigma) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i>=0 && i<x-2 && j>=0 && j<y-2 && k>=2 && k<z-2) {
        unsigned int material_index = S[k + z*(j + y*i)];
        float a = (2 * tau_sigma[0] - dt) / (2 * tau_sigma[0] + dt);
        float b = (2 * dt) / (2 * tau_sigma[0] + dt);
        Rx[k + z*(j + y*i)] = a * Rx[k + z*(j + y*i)] - b * (eta_tau_p[material_index] * Uxx[k + z*(j + y*i)] + (eta_tau_p[material_index] - 2*mu_tau_s[material_index]) * (Uyy[k + z*(j + y*i)] + Uzz[k + z*(j + y*i)]));
    }
}

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void RRyy(float dt, float* Ry, float* Uxx, float* Uyy, float* Uzz, unsigned int* S, float* eta_tau_p, float* mu_tau_s, float* tau_sigma) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i>=0 && i<x-2 && j>=0 && j<y-2 && k>=2 && k<z-2) {
        unsigned int material_index = S[k + z*(j + y*i)];
        float a = (2 * tau_sigma[0] - dt) / (2 * tau_sigma[0] + dt);
        float b = (2 * dt) / (2 * tau_sigma[0] + dt);
        Ry[k + z*(j + y*i)] = a * Ry[k + z*(j + y*i)] - b * (eta_tau_p[material_index] * Uyy[k + z*(j + y*i)] + (eta_tau_p[material_index] - 2*mu_tau_s[material_index]) * (Uxx[k + z*(j + y*i)] + Uzz[k + z*(j + y*i)]));
    }
}

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void RRzz(float dt, float* Rz, float* Uxx, float* Uyy, float* Uzz, unsigned int* S, float* eta_tau_p, float* mu_tau_s, float* tau_sigma) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i>=0 && i<x-2 && j>=0 && j<y-2 && k>=2 && k<z-2) {
        unsigned int material_index = S[k + z*(j + y*i)];
        float a = (2 * tau_sigma[0] - dt) / (2 * tau_sigma[0] + dt);
        float b = (2 * dt) / (2 * tau_sigma[0] + dt);
        Rz[k + z*(j + y*i)] = a * Rz[k + z*(j + y*i)] - b * (eta_tau_p[material_index] * Uzz[k + z*(j + y*i)] + (eta_tau_p[material_index] - 2*mu_tau_s[material_index]) * (Uxx[k + z*(j + y*i)] + Uyy[k + z*(j + y*i)]));
    }
}

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void RRxy(float dt, float* Rxy, float* Ux, float* Uy, unsigned int* S, float* mu_tau_s, float* tau_sigma) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i>=0 && i<x-2 && j>=0 && j<y-2 && k>=2 && k<z-2) {
        float Uxy = - Ux[k + z*(j+2 + y*i)] + 27 * (Ux[k + z*(j+1 + y*i)] - Ux[k + z*(j + y*i)]) + Ux[k + z*(abs(int(j)-1) + y*i)];
        float Uyx = - Uy[k+ z*(j + y*(i+2))] + 27 * (Uy[k + z*(j + y*(i+1))] - Uy[k + z*(j + y*i)]) + Uy[k + z*(j + y*abs(int(i)-1))];
        unsigned int material_index = S[k + z*(j + y*i)];
        float a = (2 * tau_sigma[0] - dt) / (2 * tau_sigma[0] + dt);
        float b = (2 * dt) / (2 * tau_sigma[0] + dt);
        Rxy[k + z*(j + y*i)] = a * Rxy[k + z*(j + y*i)] - b * (mu_tau_s[material_index] * (Uxy + Uyx));
    }
}

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void RRyz(float dt, float* Ryz, float* Uy, float* Uz, unsigned int* S, float* mu_tau_s, float* tau_sigma) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i>=0 && i<x-2 && j>=0 && j<y-2 && k>=2 && k<z-2) {
        float Uyz = - Uy[k+2+ z*(j + y*i)] + 27 * (Uy[k+1 + z*(j + y*i)] - Uy[k + z*(j + y*i)]) + Uy[k-1 + z*(j + y*i)];
        float Uzy = - Uz[k + z*(j+2 + y*i)] + 27 * (Uz[k + z*(j+1 + y*i)] - Uz[k + z*(j + y*i)]) + Uz[k + z*(abs(int(j)-1) + y*i)];
        unsigned int material_index = S[k + z*(j + y*i)];
        float a = (2 * tau_sigma[0] - dt) / (2 * tau_sigma[0] + dt);
        float b = (2 * dt) / (2 * tau_sigma[0] + dt);
        Ryz[k + z*(j + y*i)] = a * Ryz[k + z*(j + y*i)] - b * (mu_tau_s[material_index] * (Uyz + Uzy));
    }
}

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void RRxz(float dt, float* Rxz, float* Ux, float* Uz, unsigned int* S, float* mu_tau_s, float* tau_sigma) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i>=0 && i<x-2 && j>=0 && j<y-2 && k>=2 && k<z-2) {
        float Uxz = - Ux[k+2+ z*(j + y*i)] + 27 * (Ux[k+1 + z*(j + y*i)] - Ux[k + z*(j + y*i)]) + Ux[k-1 + z*(j + y*i)];
        float Uzx = - Uz[k+ z*(j + y*(i+2))] + 27 * (Uz[k + z*(j + y*(i+1))] - Uz[k + z*(j + y*i)]) + Uz[k + z*(j + y*abs(int(i)-1))];
        unsigned int material_index = S[k + z*(j + y*i)];
        float a = (2 * tau_sigma[0] - dt) / (2 * tau_sigma[0] + dt);
        float b = (2 * dt) / (2 * tau_sigma[0] + dt);
        Rxz[k + z*(j + y*i)] = a * Rxz[k + z*(j + y*i)] - b * (mu_tau_s[material_index] * (Uxz + Uzx));
    }
}

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void RPxx(float dt, float* Px, float* Uxx, float* Uyy, float* Uzz, float* Rx, unsigned int* S, float* eta_tau_p, float* mu_tau_s, float* F) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i>=0 && i<x-2 && j>=0 && j<y-2 && k>=2 && k<z-2) {
        unsigned int material_index = S[k + z*(j + y*i)];
        Px[k + z*(j + y*i)] += dt * (eta_tau_p[material_index] * Uxx[k + z*(j + y*i)] + (eta_tau_p[material_index] - 2*mu_tau_s[material_index]) * (Uyy[k + z*(j + y*i)] + Uzz[k + z*(j + y*i)]) - 0.5 * Rx[k + z*(j + y*i)] - F[k + z*(j + y*i)]);
    }
}

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void RPyy(float dt, float* Py, float* Uxx, float* Uyy, float* Uzz, float* Ry, unsigned int* S, float* eta_tau_p, float* mu_tau_s, float* F) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i>=0 && i<x-2 && j>=0 && j<y-2 && k>=2 && k<z-2) {
        unsigned int material_index = S[k + z*(j + y*i)];
        Py[k + z*(j + y*i)] += dt * (eta_tau_p[material_index] * Uyy[k + z*(j + y*i)] + (eta_tau_p[material_index] - 2*mu_tau_s[material_index]) * (Uxx[k + z*(j + y*i)] + Uzz[k + z*(j + y*i)]) - 0.5 * Ry[k + z*(j + y*i)] - F[k + z*(j + y*i)]);
    }
}

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void RPzz(float dt, float* Pz, float* Uxx, float* Uyy, float* Uzz, float* Rz, unsigned int* S, float* eta_tau_p, float* mu_tau_s, float* F) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i>=0 && i<x-2 && j>=0 && j<y-2 && k>=2 && k<z-2) {
        unsigned int material_index = S[k + z*(j + y*i)];
        Pz[k + z*(j + y*i)] += dt * (eta_tau_p[material_index] * Uzz[k + z*(j + y*i)] + (eta_tau_p[material_index] - 2*mu_tau_s[material_index]) * (Uxx[k + z*(j + y*i)] + Uyy[k + z*(j + y*i)]) - 0.5 * Rz[k + z*(j + y*i)] - F[k + z*(j + y*i)]);
    }
}

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void RPxy(float dt, float* Pxy, float* Ux, float* Uy, float* Rxy, unsigned int* S, float* mu_tau_s) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i>=0 && i<x-2 && j>=0 && j<y-2 && k>=2 && k<z-2) {
        float Uxy = - Ux[k + z*(j+2 + y*i)] + 27 * (Ux[k + z*(j+1 + y*i)] - Ux[k + z*(j + y*i)]) + Ux[k + z*(abs(int(j)-1) + y*i)];
        float Uyx = - Uy[k+ z*(j + y*(i+2))] + 27 * (Uy[k + z*(j + y*(i+1))] - Uy[k + z*(j + y*i)]) + Uy[k + z*(j + y*abs(int(i)-1))];
        unsigned int material_index = S[k + z*(j + y*i)];
        Pxy[k + z*(j + y*i)] += dt * (mu_tau_s[material_index] * (Uxy + Uyx) - 0.5 * Rxy[k + z*(j + y*i)]);
    }
}

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void RPyz(float dt, float* Pyz, float* Uy, float* Uz, float* Ryz, unsigned int* S, float* mu_tau_s) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i>=0 && i<x-2 && j>=0 && j<y-2 && k>=2 && k<z-2) {
        float Uyz = - Uy[k+2+ z*(j + y*i)] + 27 * (Uy[k+1 + z*(j + y*i)] - Uy[k + z*(j + y*i)]) + Uy[k-1 + z*(j + y*i)];
        float Uzy = - Uz[k + z*(j+2 + y*i)] + 27 * (Uz[k + z*(j+1 + y*i)] - Uz[k + z*(j + y*i)]) + Uz[k + z*(abs(int(j)-1) + y*i)];
        unsigned int material_index = S[k + z*(j + y*i)];
        Pyz[k + z*(j + y*i)] += dt * (mu_tau_s[material_index] * (Uyz + Uzy) - 0.5 * Ryz[k + z*(j + y*i)]);
    }
}

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void RPxz(float dt, float* Pxz, float* Ux, float* Uz, float* Rxz, unsigned int* S, float* mu_tau_s) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i>=0 && i<x-2 && j>=0 && j<y-2 && k>=2 && k<z-2) {
        float Uxz = - Ux[k+2+ z*(j + y*i)] + 27 * (Ux[k+1 + z*(j + y*i)] - Ux[k + z*(j + y*i)]) + Ux[k-1 + z*(j + y*i)];
        float Uzx = - Uz[k+ z*(j + y*(i+2))] + 27 * (Uz[k + z*(j + y*(i+1))] - Uz[k + z*(j + y*i)]) + Uz[k + z*(j + y*abs(int(i)-1))];
        unsigned int material_index = S[k + z*(j + y*i)];
        Pxz[k + z*(j + y*i)] += dt * (mu_tau_s[material_index] * (Uxz + Uzx) - 0.5 * Rxz[k + z*(j + y*i)]);
    }
}

template<unsigned int x, unsigned int y, unsigned int z, unsigned int N, typename T>
class SolverReflection : public Scene<x, y, z, N, T, SolverReflection<x, y, z, N, T>>{

    /// Init
    public: void Init();

    /// Step Simulation
    /// Compute the next state
    public: void Step();

};

template<unsigned int x, unsigned int y, unsigned int z, unsigned int N, typename T>
void SolverReflection<x, y, z, N, T>::Init() {
    /// Absorbing Boundary Condition
    float zeta_min = 0.95;
    float p = 1.6;

    auto Op = RBound<x, y, z, N>(zeta_min, p);
    thrust::counting_iterator<int> idxfirst(0);
    thrust::counting_iterator<int> idxlast = idxfirst + x * y * z;
    thrust::transform(idxfirst, idxlast, Scene<x, y, z, N, T, SolverReflection>::B.begin(), Scene<x, y, z, N, T, SolverReflection>::B.begin(), Op);
}

template<unsigned int x, unsigned int y, unsigned int z, unsigned int N, typename T>
void SolverReflection<x, y, z, N, T>::Step() {
// Emitter Field computing
    F<x, y, z, T><<<1, (Scene<x, y, z, N, T, SolverReflection>::emitters).size()>>>(Scene<x, y, z, N, T, SolverReflection>::Time(), thrust::raw_pointer_cast(Scene<x, y, z, N, T, SolverReflection>::emitters.data()), thrust::raw_pointer_cast(Scene<x, y, z, N, T, SolverReflection>::E.data()));

    dim3 ThreadPerBlock(4, 4, 4);
    dim3 GridDimension(int(Scene<x, y, z, N, T, SolverReflection>::X()/Scene<x, y, z, N, T, SolverReflection>::dX()) / ThreadPerBlock.x, int(Scene<x, y, z, N, T, SolverReflection>::Y()/Scene<x, y, z, N, T, SolverReflection>::dY()) / ThreadPerBlock.y, int(Scene<x, y, z, N, T, SolverReflection>::Z()/Scene<x, y, z, N, T, SolverReflection>::dZ()) / ThreadPerBlock.z);

    RUx<x, y, z><<<GridDimension, ThreadPerBlock>>>(
        Scene<x, y, z, N, T, SolverReflection>::TimeStep(),
        thrust::raw_pointer_cast(Scene<x, y, z, N, T, SolverReflection>::m_alpha.data()),
        thrust::raw_pointer_cast(Scene<x, y, z, N, T, SolverReflection>::U.x.data()),
        thrust::raw_pointer_cast(Scene<x, y, z, N, T, SolverReflection>::P.x.data()),
        thrust::raw_pointer_cast(Scene<x, y, z, N, T, SolverReflection>::P.xy.data()),
        thrust::raw_pointer_cast(Scene<x, y, z, N, T, SolverReflection>::P.xz.data()),
        thrust::raw_pointer_cast(Scene<x, y, z, N, T, SolverReflection>::m_M.data()),
        thrust::raw_pointer_cast(Scene<x, y, z, N, T, SolverReflection>::m_device_materials.inv_rho.data())
    );

    RUy<x, y, z><<<GridDimension, ThreadPerBlock>>>(
        Scene<x, y, z, N, T, SolverReflection>::TimeStep(),
        thrust::raw_pointer_cast(Scene<x, y, z, N, T, SolverReflection>::m_alpha.data()),
        thrust::raw_pointer_cast(Scene<x, y, z, N, T, SolverReflection>::U.y.data()),
        thrust::raw_pointer_cast(Scene<x, y, z, N, T, SolverReflection>::P.y.data()),
        thrust::raw_pointer_cast(Scene<x, y, z, N, T, SolverReflection>::P.xy.data()),
        thrust::raw_pointer_cast(Scene<x, y, z, N, T, SolverReflection>::P.yz.data()),
        thrust::raw_pointer_cast(Scene<x, y, z, N, T, SolverReflection>::m_M.data()),
        thrust::raw_pointer_cast(Scene<x, y, z, N, T, SolverReflection>::m_device_materials.inv_rho.data())
    );

    RUz<x, y, z><<<GridDimension, ThreadPerBlock>>>(
        Scene<x, y, z, N, T, SolverReflection>::TimeStep(),
        thrust::raw_pointer_cast(Scene<x, y, z, N, T, SolverReflection>::m_alpha.data()),
        thrust::raw_pointer_cast(Scene<x, y, z, N, T, SolverReflection>::U.z.data()),
        thrust::raw_pointer_cast(Scene<x, y, z, N, T, SolverReflection>::P.z.data()),
        thrust::raw_pointer_cast(Scene<x, y, z, N, T, SolverReflection>::P.yz.data()),
        thrust::raw_pointer_cast(Scene<x, y, z, N, T, SolverReflection>::P.xz.data()),
        thrust::raw_pointer_cast(Scene<x, y, z, N, T, SolverReflection>::m_M.data()),
        thrust::raw_pointer_cast(Scene<x, y, z, N, T, SolverReflection>::m_device_materials.inv_rho.data())
    );

    // Let each kernels finising their tasks
    cudaDeviceSynchronize();

    RUxx<x, y, z><<<GridDimension, ThreadPerBlock>>>(
        Scene<x, y, z, N, T, SolverReflection>::m_alpha[0],
        thrust::raw_pointer_cast(Scene<x, y, z, N, T, SolverReflection>::U.x.data()),
        thrust::raw_pointer_cast(Scene<x, y, z, N, T, SolverReflection>::dU.x.data())
    );

    RUyy<x, y, z><<<GridDimension, ThreadPerBlock>>>(
        Scene<x, y, z, N, T, SolverReflection>::m_alpha[1],
        thrust::raw_pointer_cast(Scene<x, y, z, N, T, SolverReflection>::U.y.data()),
        thrust::raw_pointer_cast(Scene<x, y, z, N, T, SolverReflection>::dU.y.data())
    );

    RUzz<x, y, z><<<GridDimension, ThreadPerBlock>>>(
        Scene<x, y, z, N, T, SolverReflection>::m_alpha[2],
        thrust::raw_pointer_cast(Scene<x, y, z, N, T, SolverReflection>::U.z.data()),
        thrust::raw_pointer_cast(Scene<x, y, z, N, T, SolverReflection>::dU.z.data())
    );

    // Let each kernels finising their tasks
    cudaDeviceSynchronize();

    RPxx<x, y, z><<<GridDimension, ThreadPerBlock>>>(
        Scene<x, y, z, N, T, SolverReflection>::TimeStep(),
        thrust::raw_pointer_cast(Scene<x, y, z, N, T, SolverReflection>::P.x.data()),
        thrust::raw_pointer_cast(Scene<x, y, z, N, T, SolverReflection>::dU.x.data()),
        thrust::raw_pointer_cast(Scene<x, y, z, N, T, SolverReflection>::dU.y.data()),
        thrust::raw_pointer_cast(Scene<x, y, z, N, T, SolverReflection>::dU.z.data()),
        thrust::raw_pointer_cast(Scene<x, y, z, N, T, SolverReflection>::R.x.data()),
        thrust::raw_pointer_cast(Scene<x, y, z, N, T, SolverReflection>::m_M.data()),
        thrust::raw_pointer_cast(Scene<x, y, z, N, T, SolverReflection>::m_device_materials.eta_tau_p.data()),
        thrust::raw_pointer_cast(Scene<x, y, z, N, T, SolverReflection>::m_device_materials.mu_tau_s.data()),
        thrust::raw_pointer_cast(Scene<x, y, z, N, T, SolverReflection>::E.data())
    );

    RPyy<x, y, z><<<GridDimension, ThreadPerBlock>>>(
        Scene<x, y, z, N, T, SolverReflection>::TimeStep(),
        thrust::raw_pointer_cast(Scene<x, y, z, N, T, SolverReflection>::P.y.data()),
        thrust::raw_pointer_cast(Scene<x, y, z, N, T, SolverReflection>::dU.x.data()),
        thrust::raw_pointer_cast(Scene<x, y, z, N, T, SolverReflection>::dU.y.data()),
        thrust::raw_pointer_cast(Scene<x, y, z, N, T, SolverReflection>::dU.z.data()),
        thrust::raw_pointer_cast(Scene<x, y, z, N, T, SolverReflection>::R.y.data()),
        thrust::raw_pointer_cast(Scene<x, y, z, N, T, SolverReflection>::m_M.data()),
        thrust::raw_pointer_cast(Scene<x, y, z, N, T, SolverReflection>::m_device_materials.eta_tau_p.data()),
        thrust::raw_pointer_cast(Scene<x, y, z, N, T, SolverReflection>::m_device_materials.mu_tau_s.data()),
        thrust::raw_pointer_cast(Scene<x, y, z, N, T, SolverReflection>::E.data())
    );

    RPzz<x, y, z><<<GridDimension, ThreadPerBlock>>>(
        Scene<x, y, z, N, T, SolverReflection>::TimeStep(),
        thrust::raw_pointer_cast(Scene<x, y, z, N, T, SolverReflection>::P.z.data()),
        thrust::raw_pointer_cast(Scene<x, y, z, N, T, SolverReflection>::dU.x.data()),
        thrust::raw_pointer_cast(Scene<x, y, z, N, T, SolverReflection>::dU.y.data()),
        thrust::raw_pointer_cast(Scene<x, y, z, N, T, SolverReflection>::dU.z.data()),
        thrust::raw_pointer_cast(Scene<x, y, z, N, T, SolverReflection>::R.z.data()),
        thrust::raw_pointer_cast(Scene<x, y, z, N, T, SolverReflection>::m_M.data()),
        thrust::raw_pointer_cast(Scene<x, y, z, N, T, SolverReflection>::m_device_materials.eta_tau_p.data()),
        thrust::raw_pointer_cast(Scene<x, y, z, N, T, SolverReflection>::m_device_materials.mu_tau_s.data()),
        thrust::raw_pointer_cast(Scene<x, y, z, N, T, SolverReflection>::E.data())
    );

    RPxy<x, y, z><<<GridDimension, ThreadPerBlock>>>(
        Scene<x, y, z, N, T, SolverReflection>::TimeStep(),
        thrust::raw_pointer_cast(Scene<x, y, z, N, T, SolverReflection>::P.xy.data()),
        thrust::raw_pointer_cast(Scene<x, y, z, N, T, SolverReflection>::U.x.data()),
        thrust::raw_pointer_cast(Scene<x, y, z, N, T, SolverReflection>::U.y.data()),
        thrust::raw_pointer_cast(Scene<x, y, z, N, T, SolverReflection>::R.xy.data()),
        thrust::raw_pointer_cast(Scene<x, y, z, N, T, SolverReflection>::m_M.data()),
        thrust::raw_pointer_cast(Scene<x, y, z, N, T, SolverReflection>::m_device_materials.mu_tau_s.data())
    );

    RPyz<x, y, z><<<GridDimension, ThreadPerBlock>>>(
        Scene<x, y, z, N, T, SolverReflection>::TimeStep(),
        thrust::raw_pointer_cast(Scene<x, y, z, N, T, SolverReflection>::P.yz.data()),
        thrust::raw_pointer_cast(Scene<x, y, z, N, T, SolverReflection>::U.y.data()),
        thrust::raw_pointer_cast(Scene<x, y, z, N, T, SolverReflection>::U.z.data()),
        thrust::raw_pointer_cast(Scene<x, y, z, N, T, SolverReflection>::R.yz.data()),
        thrust::raw_pointer_cast(Scene<x, y, z, N, T, SolverReflection>::m_M.data()),
        thrust::raw_pointer_cast(Scene<x, y, z, N, T, SolverReflection>::m_device_materials.mu_tau_s.data())
    );

    RPxz<x, y, z><<<GridDimension, ThreadPerBlock>>>(
        Scene<x, y, z, N, T, SolverReflection>::TimeStep(),
        thrust::raw_pointer_cast(Scene<x, y, z, N, T, SolverReflection>::P.xz.data()),
        thrust::raw_pointer_cast(Scene<x, y, z, N, T, SolverReflection>::U.x.data()),
        thrust::raw_pointer_cast(Scene<x, y, z, N, T, SolverReflection>::U.z.data()),
        thrust::raw_pointer_cast(Scene<x, y, z, N, T, SolverReflection>::R.xz.data()),
        thrust::raw_pointer_cast(Scene<x, y, z, N, T, SolverReflection>::m_M.data()),
        thrust::raw_pointer_cast(Scene<x, y, z, N, T, SolverReflection>::m_device_materials.mu_tau_s.data())
    );

    // Let each kernels finising their tasks
    cudaDeviceSynchronize();

    RRxx<x, y, z><<<GridDimension, ThreadPerBlock>>>(
        Scene<x, y, z, N, T, SolverReflection>::TimeStep(),
        thrust::raw_pointer_cast(Scene<x, y, z, N, T, SolverReflection>::R.x.data()),
        thrust::raw_pointer_cast(Scene<x, y, z, N, T, SolverReflection>::dU.x.data()),
        thrust::raw_pointer_cast(Scene<x, y, z, N, T, SolverReflection>::dU.y.data()),
        thrust::raw_pointer_cast(Scene<x, y, z, N, T, SolverReflection>::dU.z.data()),
        thrust::raw_pointer_cast(Scene<x, y, z, N, T, SolverReflection>::m_M.data()),
        thrust::raw_pointer_cast(Scene<x, y, z, N, T, SolverReflection>::m_device_materials.eta_tau_p.data()),
        thrust::raw_pointer_cast(Scene<x, y, z, N, T, SolverReflection>::m_device_materials.mu_tau_s.data()),
        thrust::raw_pointer_cast(Scene<x, y, z, N, T, SolverReflection>::m_tau_sigma.data())
    );

    RRyy<x, y, z><<<GridDimension, ThreadPerBlock>>>(
        Scene<x, y, z, N, T, SolverReflection>::TimeStep(),
        thrust::raw_pointer_cast(Scene<x, y, z, N, T, SolverReflection>::R.y.data()),
        thrust::raw_pointer_cast(Scene<x, y, z, N, T, SolverReflection>::dU.x.data()),
        thrust::raw_pointer_cast(Scene<x, y, z, N, T, SolverReflection>::dU.y.data()),
        thrust::raw_pointer_cast(Scene<x, y, z, N, T, SolverReflection>::dU.z.data()),
        thrust::raw_pointer_cast(Scene<x, y, z, N, T, SolverReflection>::m_M.data()),
        thrust::raw_pointer_cast(Scene<x, y, z, N, T, SolverReflection>::m_device_materials.eta_tau_p.data()),
        thrust::raw_pointer_cast(Scene<x, y, z, N, T, SolverReflection>::m_device_materials.mu_tau_s.data()),
        thrust::raw_pointer_cast(Scene<x, y, z, N, T, SolverReflection>::m_tau_sigma.data())
    );

    RRzz<x, y, z><<<GridDimension, ThreadPerBlock>>>(
        Scene<x, y, z, N, T, SolverReflection>::TimeStep(),
        thrust::raw_pointer_cast(Scene<x, y, z, N, T, SolverReflection>::R.z.data()),
        thrust::raw_pointer_cast(Scene<x, y, z, N, T, SolverReflection>::dU.x.data()),
        thrust::raw_pointer_cast(Scene<x, y, z, N, T, SolverReflection>::dU.y.data()),
        thrust::raw_pointer_cast(Scene<x, y, z, N, T, SolverReflection>::dU.z.data()),
        thrust::raw_pointer_cast(Scene<x, y, z, N, T, SolverReflection>::m_M.data()),
        thrust::raw_pointer_cast(Scene<x, y, z, N, T, SolverReflection>::m_device_materials.eta_tau_p.data()),
        thrust::raw_pointer_cast(Scene<x, y, z, N, T, SolverReflection>::m_device_materials.mu_tau_s.data()),
        thrust::raw_pointer_cast(Scene<x, y, z, N, T, SolverReflection>::m_tau_sigma.data())
    );

    RRxy<x, y, z><<<GridDimension, ThreadPerBlock>>>(
        Scene<x, y, z, N, T, SolverReflection>::TimeStep(),
        thrust::raw_pointer_cast(Scene<x, y, z, N, T, SolverReflection>::R.xy.data()),
        thrust::raw_pointer_cast(Scene<x, y, z, N, T, SolverReflection>::U.x.data()),
        thrust::raw_pointer_cast(Scene<x, y, z, N, T, SolverReflection>::U.y.data()),
        thrust::raw_pointer_cast(Scene<x, y, z, N, T, SolverReflection>::m_M.data()),
        thrust::raw_pointer_cast(Scene<x, y, z, N, T, SolverReflection>::m_device_materials.mu_tau_s.data()),
        thrust::raw_pointer_cast(Scene<x, y, z, N, T, SolverReflection>::m_tau_sigma.data())
    );

    RRyz<x, y, z><<<GridDimension, ThreadPerBlock>>>(
        Scene<x, y, z, N, T, SolverReflection>::TimeStep(),
        thrust::raw_pointer_cast(Scene<x, y, z, N, T, SolverReflection>::R.yz.data()),
        thrust::raw_pointer_cast(Scene<x, y, z, N, T, SolverReflection>::U.y.data()),
        thrust::raw_pointer_cast(Scene<x, y, z, N, T, SolverReflection>::U.z.data()),
        thrust::raw_pointer_cast(Scene<x, y, z, N, T, SolverReflection>::m_M.data()),
        thrust::raw_pointer_cast(Scene<x, y, z, N, T, SolverReflection>::m_device_materials.mu_tau_s.data()),
        thrust::raw_pointer_cast(Scene<x, y, z, N, T, SolverReflection>::m_tau_sigma.data())
    );

    RRxz<x, y, z><<<GridDimension, ThreadPerBlock>>>(
        Scene<x, y, z, N, T, SolverReflection>::TimeStep(),
        thrust::raw_pointer_cast(Scene<x, y, z, N, T, SolverReflection>::R.xz.data()),
        thrust::raw_pointer_cast(Scene<x, y, z, N, T, SolverReflection>::U.x.data()),
        thrust::raw_pointer_cast(Scene<x, y, z, N, T, SolverReflection>::U.z.data()),
        thrust::raw_pointer_cast(Scene<x, y, z, N, T, SolverReflection>::m_M.data()),
        thrust::raw_pointer_cast(Scene<x, y, z, N, T, SolverReflection>::m_device_materials.mu_tau_s.data()),
        thrust::raw_pointer_cast(Scene<x, y, z, N, T, SolverReflection>::m_tau_sigma.data())
    );

    // Let each kernels finising their tasks
    cudaDeviceSynchronize();

    // Finishing P update
    auto func = saxpy_functor(- Scene<x, y, z, N, T, SolverReflection>::TimeStep() * 0.5);
    thrust::transform(Scene<x, y, z, N, T, SolverReflection>::R.x.begin(), Scene<x, y, z, N, T, SolverReflection>::R.x.end(), Scene<x, y, z, N, T, SolverReflection>::P.x.begin(), Scene<x, y, z, N, T, SolverReflection>::P.x.begin(), func);
    thrust::transform(Scene<x, y, z, N, T, SolverReflection>::R.y.begin(), Scene<x, y, z, N, T, SolverReflection>::R.y.end(), Scene<x, y, z, N, T, SolverReflection>::P.y.begin(), Scene<x, y, z, N, T, SolverReflection>::P.y.begin(), func);
    thrust::transform(Scene<x, y, z, N, T, SolverReflection>::R.z.begin(), Scene<x, y, z, N, T, SolverReflection>::R.z.end(), Scene<x, y, z, N, T, SolverReflection>::P.z.begin(), Scene<x, y, z, N, T, SolverReflection>::P.z.begin(), func);
    thrust::transform(Scene<x, y, z, N, T, SolverReflection>::R.xy.begin(), Scene<x, y, z, N, T, SolverReflection>::R.xy.end(), Scene<x, y, z, N, T, SolverReflection>::P.xy.begin(), Scene<x, y, z, N, T, SolverReflection>::P.xy.begin(), func);
    thrust::transform(Scene<x, y, z, N, T, SolverReflection>::R.yz.begin(), Scene<x, y, z, N, T, SolverReflection>::R.yz.end(), Scene<x, y, z, N, T, SolverReflection>::P.yz.begin(), Scene<x, y, z, N, T, SolverReflection>::P.yz.begin(), func);
    thrust::transform(Scene<x, y, z, N, T, SolverReflection>::R.xz.begin(), Scene<x, y, z, N, T, SolverReflection>::R.xz.end(), Scene<x, y, z, N, T, SolverReflection>::P.xz.begin(), Scene<x, y, z, N, T, SolverReflection>::P.xz.begin(), func);

    cudaDeviceSynchronize();

    // Applying Absorbing Boundary Condition
    thrust::transform(Scene<x, y, z, N, T, SolverReflection>::U.x.begin(), Scene<x, y, z, N, T, SolverReflection>::U.x.end(), Scene<x, y, z, N, T, SolverReflection>::B.begin(), Scene<x, y, z, N, T, SolverReflection>::U.x.begin(), thrust::multiplies<float>());
    thrust::transform(Scene<x, y, z, N, T, SolverReflection>::U.y.begin(), Scene<x, y, z, N, T, SolverReflection>::U.y.end(), Scene<x, y, z, N, T, SolverReflection>::B.begin(), Scene<x, y, z, N, T, SolverReflection>::U.y.begin(), thrust::multiplies<float>());
    thrust::transform(Scene<x, y, z, N, T, SolverReflection>::U.z.begin(), Scene<x, y, z, N, T, SolverReflection>::U.z.end(), Scene<x, y, z, N, T, SolverReflection>::B.begin(), Scene<x, y, z, N, T, SolverReflection>::U.z.begin(), thrust::multiplies<float>());
    thrust::transform(Scene<x, y, z, N, T, SolverReflection>::P.x.begin(), Scene<x, y, z, N, T, SolverReflection>::P.x.end(), Scene<x, y, z, N, T, SolverReflection>::B.begin(), Scene<x, y, z, N, T, SolverReflection>::P.x.begin(), thrust::multiplies<float>());
    thrust::transform(Scene<x, y, z, N, T, SolverReflection>::P.y.begin(), Scene<x, y, z, N, T, SolverReflection>::P.y.end(), Scene<x, y, z, N, T, SolverReflection>::B.begin(), Scene<x, y, z, N, T, SolverReflection>::P.y.begin(), thrust::multiplies<float>());
    thrust::transform(Scene<x, y, z, N, T, SolverReflection>::P.z.begin(), Scene<x, y, z, N, T, SolverReflection>::P.z.end(), Scene<x, y, z, N, T, SolverReflection>::B.begin(), Scene<x, y, z, N, T, SolverReflection>::P.z.begin(), thrust::multiplies<float>());
    thrust::transform(Scene<x, y, z, N, T, SolverReflection>::P.xy.begin(), Scene<x, y, z, N, T, SolverReflection>::P.xy.end(), Scene<x, y, z, N, T, SolverReflection>::B.begin(), Scene<x, y, z, N, T, SolverReflection>::P.xy.begin(), thrust::multiplies<float>());
    thrust::transform(Scene<x, y, z, N, T, SolverReflection>::P.yz.begin(), Scene<x, y, z, N, T, SolverReflection>::P.yz.end(), Scene<x, y, z, N, T, SolverReflection>::B.begin(), Scene<x, y, z, N, T, SolverReflection>::P.yz.begin(), thrust::multiplies<float>());
    thrust::transform(Scene<x, y, z, N, T, SolverReflection>::P.xz.begin(), Scene<x, y, z, N, T, SolverReflection>::P.xz.end(), Scene<x, y, z, N, T, SolverReflection>::B.begin(), Scene<x, y, z, N, T, SolverReflection>::P.xz.begin(), thrust::multiplies<float>());
    thrust::transform(Scene<x, y, z, N, T, SolverReflection>::R.x.begin(), Scene<x, y, z, N, T, SolverReflection>::R.x.end(), Scene<x, y, z, N, T, SolverReflection>::B.begin(), Scene<x, y, z, N, T, SolverReflection>::R.x.begin(), thrust::multiplies<float>());
    thrust::transform(Scene<x, y, z, N, T, SolverReflection>::R.y.begin(), Scene<x, y, z, N, T, SolverReflection>::R.y.end(), Scene<x, y, z, N, T, SolverReflection>::B.begin(), Scene<x, y, z, N, T, SolverReflection>::R.y.begin(), thrust::multiplies<float>());
    thrust::transform(Scene<x, y, z, N, T, SolverReflection>::R.z.begin(), Scene<x, y, z, N, T, SolverReflection>::R.z.end(), Scene<x, y, z, N, T, SolverReflection>::B.begin(), Scene<x, y, z, N, T, SolverReflection>::R.z.begin(), thrust::multiplies<float>());
    thrust::transform(Scene<x, y, z, N, T, SolverReflection>::R.xy.begin(), Scene<x, y, z, N, T, SolverReflection>::R.xy.end(), Scene<x, y, z, N, T, SolverReflection>::B.begin(), Scene<x, y, z, N, T, SolverReflection>::R.xy.begin(), thrust::multiplies<float>());
    thrust::transform(Scene<x, y, z, N, T, SolverReflection>::R.yz.begin(), Scene<x, y, z, N, T, SolverReflection>::R.yz.end(), Scene<x, y, z, N, T, SolverReflection>::B.begin(), Scene<x, y, z, N, T, SolverReflection>::R.yz.begin(), thrust::multiplies<float>());
    thrust::transform(Scene<x, y, z, N, T, SolverReflection>::R.xz.begin(), Scene<x, y, z, N, T, SolverReflection>::R.xz.end(), Scene<x, y, z, N, T, SolverReflection>::B.begin(), Scene<x, y, z, N, T, SolverReflection>::R.xz.begin(), thrust::multiplies<float>());

    cudaDeviceSynchronize();
};
