#pragma once

#include <thrust/random.h>


template<unsigned int h, unsigned int x, unsigned int y, unsigned int z>
__global__ void Ux(float* Ux, float* Px, float* Pxy, float* Pxz, float* rho);

template<unsigned int h, unsigned int x, unsigned int y, unsigned int z>
__global__ void Uy(float* Uy, float* Py, float* Pxy, float* Pyz, float* rho);

template<unsigned int h, unsigned int x, unsigned int y, unsigned int z>
__global__ void Uz(float* Uz, float* Pz, float* Pyz, float* Pxz, float* rho);

template<unsigned int h, unsigned int x, unsigned int y, unsigned int z>
__global__ void Rxx(float* Rx, float* Ux, float* Uy, float* Uz);

template<unsigned int h, unsigned int x, unsigned int y, unsigned int z>
__global__ void Ryy(float* Ry, float* Ux, float* Uy, float* Uz);

template<unsigned int h, unsigned int x, unsigned int y, unsigned int z>
__global__ void Rzz(float* Rz, float* Ux, float* Uy, float* Uz);

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