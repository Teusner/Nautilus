#include "core/Scene.cuh"
#include "core/Material.cuh"

#include <iostream>
#include <cmath>

#include "core/FrequencyDomain.cuh"
#include "core/kernels.cuh"

#include "export.h"

#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>

#include <cuda_profiler_api.h>


int main(void) {
    constexpr unsigned int x = 100;
    constexpr unsigned int y = 100;
    constexpr unsigned int z = 100;

    constexpr float dx = 1;
    constexpr float dy = 1;
    constexpr float dz = 1;

    constexpr unsigned int N = 20;

    constexpr float dt = 1e-5;

    // FrequencyDomain
    float omega_min = 2*M_PI*2.;
    float omega_max = 2*M_PI*25.;
    std::vector<float> tau_sigma = {1 / (2*M_PI*10)};
    FrequencyDomain freq_dom(omega_min, omega_max, tau_sigma);

    Scene<x, y, z, N, SinEmitter> s(dx, dy, dz, dt, freq_dom);

    thrust::device_vector<unsigned int> s_M(x * y * z, 0);
    s.SetScene(s_M);
    s.Init();

    SinEmitter e(50, 50, 50);
    s.emitters.push_back(e);

    unsigned int a = 4000;
    for (unsigned int i = 0; i < a; i++) {
        s.Step();
        s.m_i ++;
        if (s.m_i%1000 == 0) {
            std::cout << "Time : " << s.Time() << " s" << std::endl;
        }
    }

    thrust::host_vector<float> h_Px(s.P.x);
    std::vector<float> Px(h_Px.begin(), h_Px.end());
    thrust::host_vector<float> h_Py(s.P.y);
    std::vector<float> Py(h_Py.begin(), h_Py.end());
    thrust::host_vector<float> h_Pz(s.P.z);
    std::vector<float> Pz(h_Pz.begin(), h_Pz.end());
    to_xarray("Pressure.npy", Px, Py, Pz, x, y, z);

    return 0;
}