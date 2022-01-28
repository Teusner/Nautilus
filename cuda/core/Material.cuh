#pragma once

#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/detail/config/host_device.h>
#include <iostream>

#include "FrequencyDomain.cuh"
#define N 10

struct DeviceMaterial {
    float inv_rho;
    float eta_tau_gamma_p;
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

        __host__ DeviceMaterial GetDeviceMaterial() const { return DeviceMaterial{1 / m_rho, 1, 1}; };

    private:
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

// inline DeviceMaterial Material::GetDeviceMaterial(const FrequencyDomain &fd) {
//     return DeviceMaterial {1 / m_rho, m_rho * std::pow(m_cp, 2) * fd.tau(m_Qp), m_rho * std::pow(m_cs, 2) * fd.tau(Qs)}
// }

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