#include "Scene.h"
#include "core/Material.cuh"

#include <iostream>

#define N 3
__constant__ DeviceMaterial M[N];

int main(void) {
    dim3 d(10, 10, 10);
    Scene s(d);

    s.PrintMaterials();
    
    return 0;
}