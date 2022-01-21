#include "core/Scene.cuh"
#include "core/Solver.cuh"
#include "core/Material.cuh"

#include <iostream>


int main(void) {
    constexpr unsigned int x = 100;
    constexpr unsigned int y = 100;
    constexpr unsigned int z = 100;
    const dim3 d(x, y, z);
    Scene s(d, 1, 1, 1);

    s.PrintMaterials();

    thrust::device_vector<float> s_M(d.x * d.y * d.z, 0);
    s.SetScene(s_M);
    s.AllocateMaterials(M);

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