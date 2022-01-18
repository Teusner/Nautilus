#include "kernels.cuh"
#include "Field.h"

#include <ostream>
#include <thrust/device_vector.h>

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

template<unsigned int h, unsigned int x, unsigned int y, unsigned int z>
__global__ void Ux(float* Ux, float* Px, float* Pxy, float* Pxz, float* rho) {
    unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned int j = (blockIdx.y * blockDim.y) + threadIdx.y;
    unsigned int k = (blockIdx.z * blockDim.z) + threadIdx.z;

    if (i>=2 && i<x-2 && j>=2 && j<y-2 && k>=2 && k<z-2) {
        float dPxx = - Px[i+2 + j*x + k*y] + 27 * (Px[i+1 + j*x + k*y] - Px[i + j*x + k*y]) + Px[i-1 + j*x + k*y];
        float dPxy = - Pxy[i + (j+1)*x + k*y] + 27 * (Pxy[i + j*x + k*y] - Pxy[i + (j-1)*x + k*y]) + Pxy[i + (j-2)*x + k*y];
        float dPxz = - Pxz[i + j*x + (k+1)*y] + 27 * (Pxz[i + j*x + k*y] - Pxz[i + j*x + (k-1)*y]) + Pxz[i + j*x + (k-2)*y];
        Ux[i + j*x + k*y] += 1 / (h * rho[i + j*x + k*y]) * (dPxx + dPxy + dPxz);
    }
}

template<unsigned int h, unsigned int x, unsigned int y, unsigned int z>
__global__ void Uy(float* Uy, float* Py, float* Pxy, float* Pyz, float* rho) {
    unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned int j = (blockIdx.y * blockDim.y) + threadIdx.y;
    unsigned int k = (blockIdx.z * blockDim.z) + threadIdx.z;

    if (i>=2 && i<x-2 && j>=2 && j<y-2 && k>=2 && k<z-2) {
        float dPyy = - Py[i + (j+2)*x + k*y] + 27 * (Py[i + (j+1)*x + k*y] - Py[i + j*x + k*y]) + Py[i + (j-1)*x + k*y];
        float dPxy = - Pxy[i+1 + j*x + k*y] + 27 * (Pxy[i + j*x + k*y] - Pxy[i-1 + j*x + k*y]) + Pxy[i-2 + j*x + k*y];
        float dPyz = - Pyz[i + j*x + (k+1)*y] + 27 * (Pyz[i + j*x + k*y] - Pyz[i + j*x + (k-1)*y]) + Pyz[i + j*x + (k-2)*y];
        Uy[i + j*x + k*y] += 1 / (h * rho[i + j*x + k*y]) * (dPyy + dPxy + dPyz);
    }
}

template<unsigned int h, unsigned int x, unsigned int y, unsigned int z>
__global__ void Uz(float* Uz, float* Pz, float* Pyz, float* Pxz, float* rho) {
    unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned int j = (blockIdx.y * blockDim.y) + threadIdx.y;
    unsigned int k = (blockIdx.z * blockDim.z) + threadIdx.z;

    if (i>=2 && i<x-2 && j>=2 && j<y-2 && k>=2 && k<z-2) {
        float dPzz = - Pz[i + j*x + (k+2)*y] + 27 * (Pz[i + j*x + (k+1)*y] - Pz[i + j*x + k*y]) + Pz[i + j*x + (k-1)*y];
        float dPyz = - Pyz[i+1 + j*x + k*y] + 27 * (Pyz[i + j*x + k*y] - Pyz[i-1 + j*x + k*y]) + Pyz[i-2 + j*x + k*y];
        float dPxz = - Pxz[i + (j+1)*x + k*y] + 27 * (Pxz[i + j*x + k*y] - Pxz[i + (j-1)*x + k*y]) + Pxz[i + (j-2)*x + k*y];
        Uz[i + j*x + k*y] += 1 / (h * rho[i + j*x + k*y]) * (dPzz + dPyz + dPxz);
    }
}

template<unsigned int h, unsigned int x, unsigned int y, unsigned int z>
__global__ void Rxx(float* Rx, float* Ux, float* Uy, float* Uz) {
    unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned int j = (blockIdx.y * blockDim.y) + threadIdx.y;
    unsigned int k = (blockIdx.z * blockDim.z) + threadIdx.z;

    if (i>=2 && i<x-2 && j>=2 && j<y-2 && k>=2 && k<z-2) {
        float Uxx = - Ux[i+1 + j*x + k*y] + 27 * (Ux[i + j*x + k*y] - Ux[i-1 + j*x + k*y]) + Ux[i-2 + j*x + k*y];
        float Uyy = - Uy[i + (j+1)*x + k*y] + 27 * (Uy[i + j*x + k*y] - Uy[i + (j-1)*x + k*y]) + Uy[i + (j-2)*x + k*y];
        float Uzz = - Uz[i + j*x + (k+1)*y] + 27 * (Uz[i + j*x + k*y] - Uz[i + j*x + (k-1)*y]) + Uz[i + j*x + (k-2)*y];
        // A modifier avec Material dans la constants memory ; calcul de tau_gamma_p et de tau_gamma_s dans init()
        // Rx[i + j*x + k*y] += 1 / (h * rho[i + j*x + k*y]) * (dPxx + dPxy + dPxz);
    }
}

template<unsigned int h, unsigned int x, unsigned int y, unsigned int z>
__global__ void Ryy(float* Ry, float* Ux, float* Uy, float* Uz) {
    unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned int j = (blockIdx.y * blockDim.y) + threadIdx.y;
    unsigned int k = (blockIdx.z * blockDim.z) + threadIdx.z;

    if (i>=2 && i<x-2 && j>=2 && j<y-2 && k>=2 && k<z-2) {
        float Uxx = - Ux[i+1 + j*x + k*y] + 27 * (Ux[i + j*x + k*y] - Ux[i-1 + j*x + k*y]) + Ux[i-2 + j*x + k*y];
        float Uyy = - Uy[i + (j+1)*x + k*y] + 27 * (Uy[i + j*x + k*y] - Uy[i + (j-1)*x + k*y]) + Uy[i + (j-2)*x + k*y];
        float Uzz = - Uz[i + j*x + (k+1)*y] + 27 * (Uz[i + j*x + k*y] - Uz[i + j*x + (k-1)*y]) + Uz[i + j*x + (k-2)*y];
        // A modifier avec Material dans la constants memory ; calcul de tau_gamma_p et de tau_gamma_s dans init()
        // Ry[i + j*x + k*y] += 1 / (h * rho[i + j*x + k*y]) * (dPxx + dPxy + dPxz);
    }
}

template<unsigned int h, unsigned int x, unsigned int y, unsigned int z>
__global__ void Rzz(float* Rz, float* Ux, float* Uy, float* Uz) {
    unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned int j = (blockIdx.y * blockDim.y) + threadIdx.y;
    unsigned int k = (blockIdx.z * blockDim.z) + threadIdx.z;

    if (i>=2 && i<x-2 && j>=2 && j<y-2 && k>=2 && k<z-2) {
        float Uxx = - Ux[i+1 + j*x + k*y] + 27 * (Ux[i + j*x + k*y] - Ux[i-1 + j*x + k*y]) + Ux[i-2 + j*x + k*y];
        float Uyy = - Uy[i + (j+1)*x + k*y] + 27 * (Uy[i + j*x + k*y] - Uy[i + (j-1)*x + k*y]) + Uy[i + (j-2)*x + k*y];
        float Uzz = - Uz[i + j*x + (k+1)*y] + 27 * (Uz[i + j*x + k*y] - Uz[i + j*x + (k-1)*y]) + Uz[i + j*x + (k-2)*y];
        // A modifier avec Material dans la constants memory ; calcul de tau_gamma_p et de tau_gamma_s dans init()
        // Rz[i + j*x + k*y] += 1 / (h * rho[i + j*x + k*y]) * (dPxx + dPxy + dPxz);
    }
}


int main(int argc, char** argv) {
    const float dt = 0.01;
    const int h = 1 / dt;
    constexpr dim3 Size(100, 100, 100);
    thrust::device_vector<float> rho(Size.x*Size.y*Size.z, 1000);

    PressureField P(Size);
    VelocityField U(Size);
    MemoryField R(Size);

    /// Filling Px, Pxy, Pxz
    thrust::counting_iterator<unsigned int> index_sequence_begin(0);

    thrust::transform(index_sequence_begin, index_sequence_begin + Size.x*Size.y*Size.z, P.x.begin(), prg(-1.f,1.f));
    thrust::transform(index_sequence_begin, index_sequence_begin + Size.x*Size.y*Size.z, P.xy.begin(), prg(-1.f,1.f));
    thrust::transform(index_sequence_begin, index_sequence_begin + Size.x*Size.y*Size.z, P.xz.begin(), prg(-1.f,1.f));

    dim3 ThreadPerBlock(4, 4, 4);
    dim3 GridDimension(Size.x / ThreadPerBlock.x, Size.y / ThreadPerBlock.y, Size.z / ThreadPerBlock.z);

    Ux<h, Size.x, Size.y, Size.z><<<GridDimension, ThreadPerBlock>>>(thrust::raw_pointer_cast(&U.x[0]), thrust::raw_pointer_cast(&P.x[0]), thrust::raw_pointer_cast(&P.xy[0]), thrust::raw_pointer_cast(&P.xz[0]), thrust::raw_pointer_cast(&rho[0]));
    CUDA_CHECK( cudaPeekAtLastError() );
    cudaDeviceSynchronize();

    thrust::copy(P.x.begin() + 1000, P.x.begin() + 1010, std::ostream_iterator<float>(std::cout, " "));
    std::cout << "\n";
    thrust::copy(U.x.begin() + 1000, U.x.begin() + 1010, std::ostream_iterator<float>(std::cout, " "));

    return 0;
}