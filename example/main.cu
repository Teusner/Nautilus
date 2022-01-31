#include "core/Scene.cuh"
#include "core/Material.cuh"

#include <iostream>
#include <cmath>

#include "core/Solver.cuh"
#include "core/FrequencyDomain.cuh"
#include "core/kernels.cuh"
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>

#include <cuda_profiler_api.h>


int main(void) {
    constexpr unsigned int x = 100;
    constexpr unsigned int y = 100;
    constexpr unsigned int z = 100;

    constexpr float dx = 0.1;
    constexpr float dy = 0.1;
    constexpr float dz = 0.1;

    constexpr float dt = 0.1;

    // FrequencyDomain
    float omega_min = 2*M_PI*10.;
    float omega_max = 2*M_PI*75.;
    float Q_0 = 18;
    const unsigned int n = 1;
    FrequencyDomain freq_dom(omega_min, omega_max, n);
    float tau = freq_dom.tau(Q_0);
    std::cout << "Tau_s: " << freq_dom.TauSigma()[0] << " Tau: " << tau << std::endl;

    Scene s(x, y, z, dx, dy, dz, dt, freq_dom);

    thrust::device_vector<unsigned int> s_M(x * y * z, 0);
    s.SetScene(s_M);
    s.AllocateMaterials(M);

    SinEmitter e(10, 10, 10);
    s.emitters.push_back(e);

    Solver solver;
    solver.Init(s);
    cudaProfilerStart();
    unsigned int a = 1000;
    for (unsigned int i = 0; i < a; i++) {
        solver.Step<x, y, z, SinEmitter>(s);
        s.m_i ++;
    }
    cudaProfilerStop();

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