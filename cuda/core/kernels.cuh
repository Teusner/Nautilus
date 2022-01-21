#pragma once

#include <thrust/random.h>


template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Ux(float dt, float* Ux, float* Px, float* Pxy, float* Pxz, float* rho);

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Uy(float dt, float* Uy, float* Py, float* Pxy, float* Pyz, float* rho);

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Uz(float dt, float* Uz, float* Pz, float* Pyz, float* Pxz, float* rho);

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Rxx(float dt, float* Rx, float* Ux, float* Uy, float* Uz);

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Ryy(float dt, float* Ry, float* Ux, float* Uy, float* Uz);

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Rzz(float dt, float* Rz, float* Ux, float* Uy, float* Uz);

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