#include <iostream>

#include "core/kernels.cuh"
#include "core/export.cuh"
#include "core/Solver_Reflection.cuh"

#include <thrust/device_vector.h>


int main(void) {
    constexpr unsigned int x = 50;
    constexpr unsigned int y = 50;
    constexpr unsigned int z = 50;

    constexpr unsigned int N = 5;

    /// Creating a field on which generate the boundary
    thrust::device_vector<float> d_B(x*y*z);

    /// Applying boundary on the field
    float zeta_min = 0.95;
    float p = 2;

    auto Op = RBound<x, y, z, N>(zeta_min, p);
    thrust::counting_iterator<int> idxfirst(0);
    thrust::counting_iterator<int> idxlast = idxfirst + x*y*z;
    thrust::transform(idxfirst, idxlast, d_B.begin(), d_B.begin(), Op);

    /// Exporting the boundary field
    thrust::host_vector<float> h_B(d_B);
    std::vector<float> B(h_B.begin(), h_B.end());
    to_one_xarray("Boundary.npy", B, x, y, z);

    return 0;
}