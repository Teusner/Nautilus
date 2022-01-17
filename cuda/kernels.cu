#include "Field.h"

#include <ostream>
#include <thrust/device_vector.h>

#include <thrust/random.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

template<int h>
__global__ void Ux(float* Ux, float* Px, float* Pxy, float* Pxz, float* rho, dim3 d) {
    unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned int j = (blockIdx.y * blockDim.y) + threadIdx.y;
    unsigned int k = (blockIdx.z * blockDim.z) + threadIdx.z;

    if (i>=2 && i<d.x-2 && j>=2 && j<d.y-2 && k>=2 && k<d.z-2) {
        float dPxx = - Px[i+2 + j*d.x + k*d.y] + 27 * (Px[i+1 + j*d.x + k*d.y] - Px[i + j*d.x + k*d.y]) + Px[i-1 + j*d.x + k*d.y];
        float dPxy = - Pxy[i + (j+1)*d.x + k*d.y] + 27 * (Pxy[i + j*d.x + k*d.y] - Pxy[i + (j-1)*d.x + k * d.y]) + Pxy[i + (j-2)*d.x + k *d.y];
        float dPxz = - Pxz[i + j*d.x + (k+1)*d.y] + 27 * (Pxz[i + j*d.x + k*d.y] - Pxz[i + j*d.x + (k-1)*d.y]) + Pxz[i + j*d.x + (k-2)*d.y];
        Ux[i + j*d.x + k*d.y] += 1 / (h * rho[i + j*d.x + k*d.y]) * (dPxx + dPxy + dPxz);
    }
}

template<int h>
__global__ void Uy(float* Uy, float* Py, float* Pxy, float* Pyz, float* rho, dim3 d) {
    unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned int j = (blockIdx.y * blockDim.y) + threadIdx.y;
    unsigned int k = (blockIdx.z * blockDim.z) + threadIdx.z;

    if (i>=2 && i<d.x-2 && j>=2 && j<d.y-2 && k>=2 && k<d.z-2) {
        float dPyy = - Py[i + (j+2)*d.x + k*d.y] + 27 * (Py[i + (j+1)*d.x + k*d.y] - Py[i + j*d.x + k*d.y]) + Py[i + (j-1)*d.x + k*d.y];
        float dPxy = - Pxy[i+1 + j*d.x + k*d.y] + 27 * (Pxy[i + j*d.x + k*d.y] - Pxy[i-1 + j*d.x + k*d.y]) + Pxy[i-2 + j*d.x + k*d.y];
        float dPyz = - Pyz[i + j*d.x + (k+1)*d.y] + 27 * (Pyz[i + j*d.x + k*d.y] - Pyz[i + j*d.x + (k-1)*d.y]) + Pyz[i + j*d.x + (k-2)*d.y];
        Uy[i + j*d.x + k*d.y] += 1 / (h * rho[i + j*d.x + k*d.y]) * (dPyy + dPxy + dPyz);
    }
}

template<int h>
__global__ void Uz(float* Uz, float* Pz, float* Pyz, float* Pxz, float* rho, dim3 d) {
    unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned int j = (blockIdx.y * blockDim.y) + threadIdx.y;
    unsigned int k = (blockIdx.z * blockDim.z) + threadIdx.z;

    if (i>=2 && i<d.x-2 && j>=2 && j<d.y-2 && k>=2 && k<d.z-2) {
        float dPzz = - Pz[i + j*d.x + (k+2)*d.y] + 27 * (Pz[i + j*d.x + (k+1)*d.y] - Pz[i + j*d.x + k*d.y]) + Pz[i + j*d.x + (k-1)*d.y];
        float dPyz = - Pyz[i+1 + j*d.x + k*d.y] + 27 * (Pyz[i + j*d.x + k*d.y] - Pyz[i-1 + j*d.x + k*d.y]) + Pyz[i-2 + j*d.x + k*d.y];
        float dPxz = - Pxz[i + (j+1)*d.x + k*d.y] + 27 * (Pxz[i + j*d.x + k*d.y] - Pxz[i + (j-1)*d.x + k*d.y]) + Pxz[i + (j-2)*d.x + k*d.y];
        Uz[i + j*d.x + k*d.y] += 1 / (h * rho[i + j*d.x + k*d.y]) * (dPzz + dPyz + dPxz);
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


int main(int argc, char** argv) {
    const float dt = 0.01;
    const int h = 1 / dt;
    dim3 Size(100, 100, 100);
    thrust::device_vector<float> rho(Size.x*Size.y*Size.z, 1000);

    PressureField P(Size);
    VelocityField U(Size);

    /// Filling Px, Pxy, Pxz
    thrust::counting_iterator<unsigned int> index_sequence_begin(0);

    thrust::transform(index_sequence_begin, index_sequence_begin + Size.x*Size.y*Size.z, P.x.begin(), prg(-1.f,1.f));
    thrust::transform(index_sequence_begin, index_sequence_begin + Size.x*Size.y*Size.z, P.xy.begin(), prg(-1.f,1.f));
    thrust::transform(index_sequence_begin, index_sequence_begin + Size.x*Size.y*Size.z, P.xz.begin(), prg(-1.f,1.f));

    dim3 ThreadPerBlock(4, 4, 4);
    dim3 GridDimension(Size.x / ThreadPerBlock.x, Size.y / ThreadPerBlock.y, Size.z / ThreadPerBlock.z);

    Ux<h><<<GridDimension, ThreadPerBlock>>>(thrust::raw_pointer_cast(&U.x[0]), thrust::raw_pointer_cast(&P.x[0]), thrust::raw_pointer_cast(&P.xy[0]), thrust::raw_pointer_cast(&P.xz[0]), thrust::raw_pointer_cast(&rho[0]), Size);
    CUDA_CHECK( cudaPeekAtLastError() );
    cudaDeviceSynchronize();

    thrust::copy(P.x.begin() + 1000, P.x.begin() + 1020, std::ostream_iterator<float>(std::cout, " "));
    std::cout << "\n";
    thrust::copy(U.x.begin() + 1000, U.x.begin() + 1020, std::ostream_iterator<float>(std::cout, " "));

    return 0;
}