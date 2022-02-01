#include "kernels.cuh"
#include "Field.cuh"
#include "Material.cuh"
#include "Scene.cuh"

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