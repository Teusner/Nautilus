#include "kernels.cuh"
#include "Field.cuh"
#include "Material.cuh"
#include "Scene.cuh"

#include <ostream>
#include <thrust/device_vector.h>

#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>

#define N 10

__constant__ DeviceMaterial M[N];


#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Ux(float dt, float* Ux, float* Px, float* Pxy, float* Pxz, float* S) {
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
__global__ void Uy(float dt, float* Uy, float* Py, float* Pxy, float* Pyz, float* S) {
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
__global__ void Uz(float dt, float* Uz, float* Pz, float* Pyz, float* Pxz, float* S) {
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
__global__ void Rxx(float dt, float* Rx, float* Ux, float* Uy, float* Uz, float* S, float tau_sigma) {
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
        Rx[i + j*x + k*y] += a * Rx[i + j*x + k*y] - b * M[material_index].eta_tau_gamma_p * (Uxx + Uyy + Uzz) - 2 * M[material_index].mu_tau_gamma_s * (Uyy + Uzz);
    }
}

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Ryy(float dt, float* Ry, float* Ux, float* Uy, float* Uz, float* S, float tau_sigma) {
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
__global__ void Rzz(float dt, float* Rz, float* Ux, float* Uy, float* Uz, float* S, float tau_sigma) {
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
__global__ void Rxy(float dt, float* Rxy, float* Ux, float* Uy, float* S, float tau_sigma) {
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
__global__ void Ryz(float dt, float* Ryz, float* Uy, float* Uz, float* S, float tau_sigma) {
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
__global__ void Rxz(float dt, float* Rxz, float* Ux, float* Uz, float* S, float tau_sigma) {
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
__global__ void Pxx(float dt, float* Px, float* Ux, float* Uy, float* Uz, float* Rx, float* S) {
    unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned int j = (blockIdx.y * blockDim.y) + threadIdx.y;
    unsigned int k = (blockIdx.z * blockDim.z) + threadIdx.z;

    if (i>=2 && i<x-2 && j>=2 && j<y-2 && k>=2 && k<z-2) {
        float Uxx = - Ux[i+1 + j*x + k*y] + 27 * (Ux[i + j*x + k*y] - Ux[i-1 + j*x + k*y]) + Ux[i-2 + j*x + k*y];
        float Uyy = - Uy[i + (j+1)*x + k*y] + 27 * (Uy[i + j*x + k*y] - Uy[i + (j-1)*x + k*y]) + Uy[i + (j-2)*x + k*y];
        float Uzz = - Uz[i + j*x + (k+1)*y] + 27 * (Uz[i + j*x + k*y] - Uz[i + j*x + (k-1)*y]) + Uz[i + j*x + (k-2)*y];
        unsigned int material_index = S[i + j*x + k*y];
        Px[i + j*x + k*y] += dt * M[material_index].eta_tau_gamma_p * (Uxx + Uyy + Uzz) - 2 * M[material_index].mu_tau_gamma_s * (Uyy + Uzz) + Rx[i + j*x + k*y];
    }
}

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Pyy(float dt, float* Py, float* Ux, float* Uy, float* Uz, float* Ry, float* S) {
    unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned int j = (blockIdx.y * blockDim.y) + threadIdx.y;
    unsigned int k = (blockIdx.z * blockDim.z) + threadIdx.z;

    if (i>=2 && i<x-2 && j>=2 && j<y-2 && k>=2 && k<z-2) {
        float Uxx = - Ux[i+1 + j*x + k*y] + 27 * (Ux[i + j*x + k*y] - Ux[i-1 + j*x + k*y]) + Ux[i-2 + j*x + k*y];
        float Uyy = - Uy[i + (j+1)*x + k*y] + 27 * (Uy[i + j*x + k*y] - Uy[i + (j-1)*x + k*y]) + Uy[i + (j-2)*x + k*y];
        float Uzz = - Uz[i + j*x + (k+1)*y] + 27 * (Uz[i + j*x + k*y] - Uz[i + j*x + (k-1)*y]) + Uz[i + j*x + (k-2)*y];
        unsigned int material_index = S[i + j*x + k*y];
        Py[i + j*x + k*y] += dt * M[material_index].eta_tau_gamma_p * (Uxx + Uyy + Uzz) - 2 * M[material_index].mu_tau_gamma_s * (Uxx + Uzz) + Ry[i + j*x + k*y];
    }
}

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Pzz(float dt, float* Pz, float* Ux, float* Uy, float* Uz, float* Rz, float* S) {
    unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned int j = (blockIdx.y * blockDim.y) + threadIdx.y;
    unsigned int k = (blockIdx.z * blockDim.z) + threadIdx.z;

    if (i>=2 && i<x-2 && j>=2 && j<y-2 && k>=2 && k<z-2) {
        float Uxx = - Ux[i+1 + j*x + k*y] + 27 * (Ux[i + j*x + k*y] - Ux[i-1 + j*x + k*y]) + Ux[i-2 + j*x + k*y];
        float Uyy = - Uy[i + (j+1)*x + k*y] + 27 * (Uy[i + j*x + k*y] - Uy[i + (j-1)*x + k*y]) + Uy[i + (j-2)*x + k*y];
        float Uzz = - Uz[i + j*x + (k+1)*y] + 27 * (Uz[i + j*x + k*y] - Uz[i + j*x + (k-1)*y]) + Uz[i + j*x + (k-2)*y];
        unsigned int material_index = S[i + j*x + k*y];
        Pz[i + j*x + k*y] += dt * M[material_index].eta_tau_gamma_p * (Uxx + Uyy + Uzz) - 2 * M[material_index].mu_tau_gamma_s * (Uxx + Uyy) + Rz[i + j*x + k*y];
    }
}

template<unsigned int x, unsigned int y, unsigned int z>
__global__ void Pxy(float dt, float* Pxy, float* Ux, float* Uy, float* Rxy, float* S) {
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
__global__ void Pyz(float dt, float* Pyz, float* Uy, float* Uz, float* Ryz, float* S) {
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
__global__ void Pxz(float dt, float* Pxz, float* Ux, float* Uz, float* Rxz, float* S) {
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

// int main(int argc, char** argv) {
//     const float dt = 0.01;
//     constexpr dim3 Size(100, 100, 100);
//     thrust::device_vector<float> rho(Size.x*Size.y*Size.z, 1000);

//     PressureField P(Size);
//     VelocityField U(Size);
//     MemoryField R(Size);

//     // Scene
//     Scene scene(Size, 1, 1, 1);
//     thrust::device_vector<float> M_s(Size.x * Size.y * Size.z, 0);
//     // scene.SetScene(M_s);
//     scene.AllocateMaterials(M);

//     /// Filling Px, Pxy, Pxz
//     thrust::counting_iterator<unsigned int> index_sequence_begin(0);

//     thrust::transform(index_sequence_begin, index_sequence_begin + Size.x*Size.y*Size.z, P.x.begin(), prg(-1.f,1.f));
//     thrust::transform(index_sequence_begin, index_sequence_begin + Size.x*Size.y*Size.z, P.xy.begin(), prg(-1.f,1.f));
//     thrust::transform(index_sequence_begin, index_sequence_begin + Size.x*Size.y*Size.z, P.xz.begin(), prg(-1.f,1.f));

//     std::cout << "Initialisation\n";
//     std::cout << "P  : ";
//     thrust::copy(P.x.begin() + 1000, P.x.begin() + 1010, std::ostream_iterator<float>(std::cout, " "));
//     std::cout << "\nPxy : ";
//     thrust::copy(P.xy.begin() + 1000, P.xy.begin() + 1010, std::ostream_iterator<float>(std::cout, " "));
//     std::cout << "\nUx : ";
//     thrust::copy(U.x.begin() + 1000, U.x.begin() + 1010, std::ostream_iterator<float>(std::cout, " "));
//     std::cout << "\nRx : ";
//     thrust::copy(R.x.begin() + 1000, R.x.begin() + 1010, std::ostream_iterator<float>(std::cout, " "));
//     std::cout << "\nRxy : ";
//     thrust::copy(R.xy.begin() + 1000, R.xy.begin() + 1010, std::ostream_iterator<float>(std::cout, " "));
//     std::cout << "\n";

//     dim3 ThreadPerBlock(4, 4, 4);
//     dim3 GridDimension(Size.x / ThreadPerBlock.x, Size.y / ThreadPerBlock.y, Size.z / ThreadPerBlock.z);

//     Ux<Size.x, Size.y, Size.z><<<GridDimension, ThreadPerBlock>>>(dt, thrust::raw_pointer_cast(&U.x[0]), thrust::raw_pointer_cast(&P.x[0]), thrust::raw_pointer_cast(&P.xy[0]), thrust::raw_pointer_cast(&P.xz[0]), thrust::raw_pointer_cast(&M_s[0]));
//     CUDA_CHECK( cudaPeekAtLastError() );
//     cudaDeviceSynchronize();

//     float tau_sigma = 1;
//     Rxx<Size.x, Size.y, Size.z><<<GridDimension, ThreadPerBlock>>>(dt, thrust::raw_pointer_cast(&R.x[0]), thrust::raw_pointer_cast(&U.x[0]), thrust::raw_pointer_cast(&U.y[0]), thrust::raw_pointer_cast(&U.z[0]), thrust::raw_pointer_cast(&M_s[0]), tau_sigma);
//     CUDA_CHECK( cudaPeekAtLastError() );
//     cudaDeviceSynchronize();

//     Rxy<Size.x, Size.y, Size.z><<<GridDimension, ThreadPerBlock>>>(dt, thrust::raw_pointer_cast(&R.xy[0]), thrust::raw_pointer_cast(&U.x[0]), thrust::raw_pointer_cast(&U.y[0]), thrust::raw_pointer_cast(&M_s[0]), tau_sigma);
//     CUDA_CHECK( cudaPeekAtLastError() );
//     cudaDeviceSynchronize();

//     Pxx<Size.x, Size.y, Size.z><<<GridDimension, ThreadPerBlock>>>(dt, thrust::raw_pointer_cast(&P.x[0]), thrust::raw_pointer_cast(&U.x[0]), thrust::raw_pointer_cast(&U.y[0]), thrust::raw_pointer_cast(&U.z[0]), thrust::raw_pointer_cast(&R.x[0]), thrust::raw_pointer_cast(&M_s[0]));
//     CUDA_CHECK( cudaPeekAtLastError() );
//     cudaDeviceSynchronize();

//     Pxy<Size.x, Size.y, Size.z><<<GridDimension, ThreadPerBlock>>>(dt, thrust::raw_pointer_cast(&P.xy[0]), thrust::raw_pointer_cast(&U.x[0]), thrust::raw_pointer_cast(&U.y[0]), thrust::raw_pointer_cast(&R.xy[0]), thrust::raw_pointer_cast(&M_s[0]));
//     CUDA_CHECK( cudaPeekAtLastError() );
//     cudaDeviceSynchronize();

//     std::cout << "First Iteration\n";
//     std::cout << "P  : ";
//     thrust::copy(P.x.begin() + 1000, P.x.begin() + 1010, std::ostream_iterator<float>(std::cout, " "));
//     std::cout << "\nPxy : ";
//     thrust::copy(P.xy.begin() + 1000, P.xy.begin() + 1010, std::ostream_iterator<float>(std::cout, " "));
//     std::cout << "\nUx  : ";
//     thrust::copy(U.x.begin() + 1000, U.x.begin() + 1010, std::ostream_iterator<float>(std::cout, " "));
//     std::cout << "\nRx : ";
//     thrust::copy(R.x.begin() + 1000, R.x.begin() + 1010, std::ostream_iterator<float>(std::cout, " "));
//     std::cout << "\nRxy : ";
//     thrust::copy(R.xy.begin() + 1000, R.xy.begin() + 1010, std::ostream_iterator<float>(std::cout, " "));

//     return 0;
// }