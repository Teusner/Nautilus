#pragma once

#include <thrust/random.h>


template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Ux(float dt, float* Ux, float* Px, float* Pxy, float* Pxz, float* S);

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Uy(float dt, float* Uy, float* Py, float* Pxy, float* Pyz, float* S);

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Uz(float dt, float* Uz, float* Pz, float* Pyz, float* Pxz, float* S);

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Rxx(float dt, float* Rx, float* Ux, float* Uy, float* Uz, float* S, float tau_sigma);

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Ryy(float dt, float* Ry, float* Ux, float* Uy, float* Uz, float* S, float tau_sigma);

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Rzz(float dt, float* Rz, float* Ux, float* Uy, float* Uz, float* S, float tau_sigma);

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Rxy(float dt, float* Rxy, float* Ux, float* Uy, float* S, float tau_sigma);

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Ryz(float dt, float* Ryz, float* Uy, float* Uz, float* S, float tau_sigma);

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Rxz(float dt, float* Rxz, float* Ux, float* Uz, float* S, float tau_sigma);

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Pxx(float dt, float* Px, float* Ux, float* Uy, float* Uz, float* Rx, float* S);

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Pyy(float dt, float* Py, float* Ux, float* Uy, float* Uz, float* Rx, float* S);

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Pzz(float dt, float* Pz, float* Ux, float* Uy, float* Uz, float* Rz, float* S);

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Pxy(float dt, float* Pxy, float* Ux, float* Uy, float* Rxy, float* S);

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Pyz(float dt, float* Pyz, float* Uy, float* Uz, float* Ryz, float* S);

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Pxz(float dt, float* Pxz, float* Ux, float* Uz, float* Rxz, float* S);

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