#include "Scene.h"
#include "Solver.h"
#include "core/Material.cuh"

#include <iostream>

#define N 3
__constant__ DeviceMaterial M[N];

int main(void) {
    dim3 d(10, 10, 10);
    Scene s(d);

    s.PrintMaterials();

    std::vector<float> M(d.x * d.y * d.z, 0);
    s.SetScene(M);

    Solver solver(s);


    
    return 0;
}