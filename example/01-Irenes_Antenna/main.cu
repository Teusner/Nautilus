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
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>

#include <cuda_profiler_api.h>

#include <cxxopts.hpp>
#include <string>


int main(int argc, char *argv[]) {
    // Parsing args
    cxxopts::Options options("05-video", "Video generation of boat's enclosing state using sensors");

    options.add_options()
        ("p,path", "Output path", cxxopts::value<std::string>())
        ("h,help", "Print usage")
    ;
    auto result = options.parse(argc, argv);

    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        std::exit(EXIT_SUCCESS);
    }

    std::string p = (result["path"].as<std::string>()).empty() ? std::string("Pressure.npy") : result["path"].as<std::string>();

    std::cout << "Saving in : " << p << std::endl;

    constexpr unsigned int x = 1;
    constexpr unsigned int y = 1;
    constexpr unsigned int z = 1;

    constexpr float dx = 0.001;
    constexpr float dy = 0.001;
    constexpr float dz = 0.001;

    constexpr unsigned int N = 20;

    constexpr float dt = 1e-6;

    // FrequencyDomain
    float omega_min = 2*M_PI*100000.;
    float omega_max = 2*M_PI*110000.;
    std::vector<float> tau_sigma = {1 / (2*M_PI*106800)};
    FrequencyDomain freq_dom(omega_min, omega_max, tau_sigma);
    std::cout << int(x/dx) * int(y/dy) * int(z/dz) << std::endl;

    /// Solver
    Scene<int(x/dx), int(y/dy), int(z/dz), N, SinEmitter, SolverReflection<int(x/dx), int(y/dy), int(z/dz), N, SinEmitter>> s(dx, dy, dz, dt, freq_dom);

    thrust::device_vector<unsigned int> s_M(int(x/dx) * int(y/dy) * int(z/dz), 0);
    s.SetScene(s_M);
    s.Init();

    // Emitter
    float dim = 0.055;
    float frequency = 106800;

    for (int i=0; i<int(dim/dx); ++i) {
        for (int j=0; j<int(dim/dy); ++j) {
            SinEmitter e(i, j, N+2, 1.f, frequency);
            s.emitters.push_back(e);
        }
    }

    unsigned int a = 1000;
    for (unsigned int i = 0; i < a; ++i) {
        s.Step();
        if (s.m_i%1 == 0) {
            std::cout << "Time : " << s.Time() << " s" << std::endl;
        }
    }


    thrust::host_vector<float> h_Px(s.P.x);
    thrust::host_vector<float> h_Py(s.P.y);
    thrust::host_vector<float> h_Pz(s.P.z);

    for (unsigned int i = 0; i < 10; ++i) {
        s.Step();
        thrust::host_vector<float> ht_Px(s.P.x);
        thrust::transform(h_Px.begin(), h_Px.end(), ht_Px.begin(), ht_Px.begin(), thrust::plus<float>());
        thrust::host_vector<float> ht_Py(s.P.y);
        thrust::transform(h_Py.begin(), h_Py.end(), ht_Py.begin(), ht_Py.begin(), thrust::plus<float>());
        thrust::host_vector<float> ht_Pz(s.P.z);
        thrust::transform(h_Pz.begin(), h_Pz.end(), ht_Pz.begin(), ht_Pz.begin(), thrust::plus<float>());
    }

    std::vector<float> Px(h_Px.begin(), h_Px.end());
    std::vector<float> Py(h_Py.begin(), h_Py.end());
    std::vector<float> Pz(h_Pz.begin(), h_Pz.end());
    to_xarray<int(x/dx), int(y/dy), int(z/dz)>(p, Px, Py, Pz);

    return 0;
}