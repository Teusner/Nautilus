#include "core/Scene.cuh"
#include "core/Material.cuh"

#include <iostream>
#include <cmath>
#include <memory>

#include "core/FrequencyDomain.cuh"
#include "core/kernels.cuh"

// #include "core/export.cuh"
#include "core/Solver_Reflection.cuh"

#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>

#include <cuda_profiler_api.h>


int main(void) {
    constexpr unsigned int x = 1;
    constexpr unsigned int y = 1;
    constexpr unsigned int z = 1;

    constexpr float dx = 0.05;
    constexpr float dy = 0.05;
    constexpr float dz = 0.05;

    constexpr unsigned int N = 3;

    constexpr float dt = 1e-6;

    // FrequencyDomain
    float omega_min = 2*M_PI*95000.;
    float omega_max = 2*M_PI*105000.;
    std::vector<float> tau_sigma = {1 / (2*M_PI*100000)};
    FrequencyDomain freq_dom(omega_min, omega_max, tau_sigma);
    std::cout << int(x/dx) * int(y/dy) * int(z/dz) << std::endl;

    /// Solver
    Scene<int(x/dx), int(y/dy), int(z/dz), N, SinEmitter, SolverReflection<int(x/dx), int(y/dy), int(z/dz), N, SinEmitter>> s(dx, dy, dz, dt, freq_dom);

    thrust::device_vector<unsigned int> s_M(int(x/dx) * int(y/dy) * int(z/dz), 0);
    s.SetScene(s_M);
    s.Init();

    for (int i=0.2*int(x/dx); i<0.8*int(x/dx); ++i) {
        for (int j=0.2*int(y/dy); j<0.8*int(y/dy); ++j) {
            SinEmitter e(i, j, 5, 1.f, 100000);
            s.emitters.push_back(e);
        }
    }

    unsigned int a = 200;
    for (unsigned int i = 0; i < a; ++i) {
        s.Step();
        if (s.m_i%10 == 0) {
            std::cout << "Time : " << s.Time() << " s" << std::endl;
        }
    }

    thrust::host_vector<float> h_Px(s.P.x);
    std::vector<float> Px(h_Px.begin(), h_Px.end());
    thrust::host_vector<float> h_Py(s.P.y);
    std::vector<float> Py(h_Py.begin(), h_Py.end());
    thrust::host_vector<float> h_Pz(s.P.z);
    std::vector<float> Pz(h_Pz.begin(), h_Pz.end());
    to_xarray<int(x/dx), int(y/dy), int(z/dz)>("Pressure.npy", Px, Py, Pz);

    return 0;
}