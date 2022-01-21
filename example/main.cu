#include "core/Scene.cuh"
#include "core/Solver.cuh"
#include "core/Material.cuh"

#include <iostream>

#include "core/kernels.cuh"
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>


int main(void) {
    float dt = 0.1;
    constexpr unsigned int x = 100;
    constexpr unsigned int y = 100;
    constexpr unsigned int z = 100;
    const dim3 d(x, y, z);
    Scene s(d, 1, 1, 1, dt);

    s.PrintMaterials();

    thrust::device_vector<float> s_M(d.x * d.y * d.z, 0);
    s.SetScene(s_M);
    s.AllocateMaterials(M);

    thrust::counting_iterator<unsigned int> index_sequence_begin(0);
    thrust::transform(index_sequence_begin, index_sequence_begin + x*y*z, s.P.x.begin(), prg(-1.f,1.f));
    thrust::transform(index_sequence_begin, index_sequence_begin + x*y*z, s.P.xy.begin(), prg(-1.f,1.f));
    thrust::transform(index_sequence_begin, index_sequence_begin + x*y*z, s.P.xz.begin(), prg(-1.f,1.f));

    Solver solver;
    solver.Step<x, y, z>(s);

    std::cout << "First Iteration\n";
    std::cout << "P  : ";
    thrust::copy(s.P.x.begin() + 1000, s.P.x.begin() + 1010, std::ostream_iterator<float>(std::cout, " "));
    std::cout << "\nPxy : ";
    thrust::copy(s.P.xy.begin() + 1000, s.P.xy.begin() + 1010, std::ostream_iterator<float>(std::cout, " "));
    std::cout << "\nUx  : ";
    thrust::copy(s.U.x.begin() + 1000, s.U.x.begin() + 1010, std::ostream_iterator<float>(std::cout, " "));
    std::cout << "\nRx : ";
    thrust::copy(s.R.x.begin() + 1000, s.R.x.begin() + 1010, std::ostream_iterator<float>(std::cout, " "));
    std::cout << "\nRxy : ";
    thrust::copy(s.R.xy.begin() + 1000, s.R.xy.begin() + 1010, std::ostream_iterator<float>(std::cout, " "));
    
    return 0;
}