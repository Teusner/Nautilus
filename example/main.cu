#include "core/Scene.cuh"
#include "core/Solver.cuh"
#include "core/Material.cuh"

#include <iostream>

#define N 3
__constant__ DeviceMaterial M[N];

int main(void) {
    const unsigned int x = 10;
    const unsigned int y = 10;
    const unsigned int z = 10;
    const dim3 d(x, y, z);
    Scene s(d, 1, 1, 1);

    s.PrintMaterials();

    thrust::device_vector<float> s_M(d.x * d.y * d.z, 0);
    s.SetScene(s_M);
    s.AllocateMaterials(M);

    Solver solver;
    solver.Step<x, y, z>(s);
    
    return 0;
}