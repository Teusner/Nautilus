#pragma once

#include <cmath>

#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/detail/config/host_device.h>
#include <iostream>

#include "FrequencyDomain.cuh"
#define N 10

struct DeviceMaterial {
    float inv_rho;
    float eta_tau_epsilon_p;
    float eta_tau_gamma_p;
    float mu_tau_epsilon_s;
    float mu_tau_gamma_s;
};


class Material {
    public:
        __host__ Material() : Material(1000, 1500, 100) {};
        __host__ Material(float rho, float cp, float Qp) : m_rho(rho), m_cp(cp), m_Qp(Qp) {};
        __host__ ~Material() = default;

        __host__ float Rho() const { return m_rho; };
        __host__ float Cp() const { return m_cp; };
        __host__ float Qp() const { return m_Qp; };
        __host__ float Cs() const { return m_cs; };
        __host__ float Qs() const { return m_Qs; };

        __host__ void CopyToConstant(const void* symbol, unsigned int index) const;

        __host__ DeviceMaterial GetDeviceMaterial(unsigned int l, std::vector<float> tau_sigma, FrequencyDomain fd) const;

    private:
        /// Qp and Qs could be set for each SLS in a Material
        /// Qp and Qs become std::vector of size L
        float m_rho;
        float m_cp;
        float m_Qp;
        float m_cs = 0;
        float m_Qs = 0;
};

std::ostream &operator<<(std::ostream &os, const Material &m);
std::ostream &operator<<(std::ostream &os, const Material *m);


/// Implementation

/// Really necessary ?
inline void Material::CopyToConstant(const void* symbol, unsigned int index) const {
    // Copying one material on constant memory
    DeviceMaterial *temp_h_m = (DeviceMaterial*) malloc(sizeof(DeviceMaterial) * N);
    cudaMemcpyFromSymbol(temp_h_m, symbol, sizeof(DeviceMaterial)*N);

    // Filling the i-th DeviceMaterial
    temp_h_m[index].inv_rho = 1 / m_rho;
    temp_h_m[index].eta_tau_gamma_p = 1.2 - 1;
    temp_h_m[index].mu_tau_gamma_s = 1.2 - 1;

    cudaMemcpyToSymbol(symbol, temp_h_m, sizeof(DeviceMaterial)*N);

    free(temp_h_m);
}

inline DeviceMaterial Material::GetDeviceMaterial(unsigned int l, std::vector<float> tau_sigma, FrequencyDomain fd) const {
    /// Tau Sigma showing
    std::cout << "L: " << l << std::endl;
    std::cout << "Tau Sigma: ";
    std::copy(std::begin(tau_sigma), std::begin(tau_sigma), std::ostream_iterator<float>(std::cout, " "));
    std::cout << std::endl;

    /// Tau epsilon p computing
    std::vector<float> tau_p;
    std::transform(std::begin(tau_sigma), std::end(tau_sigma), std::back_inserter(tau_p), [=] (float t_s) { return fd.tau(m_Qp) / t_s; });
    float tau_epsilon_p = std::accumulate(std::begin(tau_p), std::end(tau_p), 0.f, std::plus<float>());

    // std::copy(std::begin(tau_p), std::end(tau_p), std::ostream_iterator<float>(std::cout, " "));
    std::cout << std::endl;

    /// Tau epsilon s computing
    std::vector<float> tau_s;
    std::transform(std::begin(tau_sigma), std::end(tau_sigma), std::back_inserter(tau_s), [=] (float t_s) { return fd.tau(m_Qs) / t_s; });
    float tau_epsilon_s = std::accumulate(std::begin(tau_s), std::end(tau_s), 0, std::plus<float>());

    float tau_gamma_p = 1 - fd.l() + tau_epsilon_p;
    float tau_gamma_s = 1 - fd.l() + tau_epsilon_s;

    std::cout << tau_gamma_p << " " << tau_gamma_s << std::endl;

    return DeviceMaterial {1 / m_rho, m_rho * powf(m_cp, 2.) * tau_epsilon_p, m_rho * powf(m_cp, 2.) * tau_gamma_p, m_rho * powf(m_cs, 2) * tau_epsilon_s, m_rho * powf(m_cs, 2) * tau_gamma_s};
}

inline std::ostream &operator<<(std::ostream &os, const Material &m) {
    return os << "{rho: " << m.Rho() << ", P: ["
                        << m.Cp() << ", " << m.Qp() << "], S: ["
                        << m.Cs() << ", " << m.Qs() << "]}";
}

inline std::ostream &operator<<(std::ostream &os, const Material *m) {
    return os << "{rho: " << m->Rho() << ", P: ["
                        << m->Cp() << ", " << m->Qp() << "], S: ["
                        << m->Cs() << ", " << m->Qs() << "]}";
}