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
    constexpr unsigned int z = 5;

    constexpr float dx = 1;
    constexpr float dy = 1;
    constexpr float dz = 1;

    constexpr float dt = 1e-5;

    // FrequencyDomain
    float omega_min = 2*M_PI*2.;
    float omega_max = 2*M_PI*25.;
    std::vector<float> tau_sigma = {1 / (2*M_PI*10)};
    FrequencyDomain freq_dom(omega_min, omega_max, tau_sigma);

    Scene s(x, y, z, dx, dy, dz, dt, freq_dom);

    thrust::device_vector<unsigned int> s_M(x * y * z, 0);
    s.SetScene(s_M);
    s.Init();

    SinEmitter e(50, 50, 2);
    s.emitters.push_back(e);

    unsigned int a = 500;
    for (unsigned int i = 0; i < a; i++) {
        s.Step<x, y, z, SinEmitter>();
        s.m_i ++;
        if (s.m_i%1000 == 0) {
            std::cout << "Time : " << s.Time() << " s" << std::endl;
        }
    }

    thrust::host_vector<float> P = s.P.x;
    std::vector<float> vec(P.begin(), P.end());
    std::vector<std::size_t> shape = {x, y, z};
    to_xarray("Pressure.npy", vec, shape);

    // std::cout << "P  : ";
    // float Px_sum = thrust::reduce(s.P.x.begin(), s.P.x.end(), 0.f);
    // std::cout << "[" << Px_sum << "] ";
    // thrust::copy(s.P.x.begin() + 1000, s.P.x.begin() + 1015, std::ostream_iterator<float>(std::cout, " "));
    // std::cout << "\nPxy : ";
    // float Pxy_sum = thrust::reduce(s.P.xy.begin(), s.P.xy.end(), 0.f);
    // std::cout << "[" << Pxy_sum << "] ";
    // thrust::copy(s.P.xy.begin() + 1000, s.P.xy.begin() + 1015, std::ostream_iterator<float>(std::cout, " "));
    // std::cout << "\nUx  : ";
    // float Ux_sum = thrust::reduce(s.U.x.begin(), s.U.x.end(), 0.f);
    // std::cout << "[" << Ux_sum << "] ";
    // thrust::copy(s.U.x.begin() + 1000, s.U.x.begin() + 1015, std::ostream_iterator<float>(std::cout, " "));
    // std::cout << "\nUxx  : ";
    // float Uxx_sum = thrust::reduce(s.dU.x.begin(), s.dU.x.end(), 0.f);
    // std::cout << "[" << Uxx_sum << "] ";
    // thrust::copy(s.dU.x.begin() + 1000, s.dU.x.begin() + 1015, std::ostream_iterator<float>(std::cout, " "));
    // std::cout << "\nRx : ";
    // float Rx_sum = thrust::reduce(s.R.x.begin(), s.R.x.end(), 0.f);
    // std::cout << "[" << Rx_sum << "] ";
    // thrust::copy(s.R.x.begin() + 1000, s.R.x.begin() + 1015, std::ostream_iterator<float>(std::cout, " "));
    // std::cout << "\nRxy : ";
    // float Rxy_sum = thrust::reduce(s.R.xy.begin(), s.R.xy.end(), 0.f);
    // std::cout << "[" << Rxy_sum << "] ";
    // thrust::copy(s.R.xy.begin() + 1000, s.R.xy.begin() + 1015, std::ostream_iterator<float>(std::cout, " "));
    
    return 0;
}