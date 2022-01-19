#include "Material.cuh"

#include <thrust/device_malloc.h>
#include <thrust/device_vector.h>

#include <stdio.h>

#define N 10


void Material::CopyToConstant(const void* symbol, unsigned int index) const {
    // Copying one material on constant memory
    DeviceMaterial *temp_h_m = (DeviceMaterial*) malloc(sizeof(DeviceMaterial) * N);
    cudaMemcpyFromSymbol(temp_h_m, symbol, sizeof(DeviceMaterial)*N);

    // Filling the i-th DeviceMaterial
    temp_h_m[index].inv_rho = 1 / m_rho;
    temp_h_m[index].eta_tau_epsilon_p = 1;
    temp_h_m[index].mu_tau_epsilon_s = 1;

    cudaMemcpyToSymbol(symbol, temp_h_m, sizeof(DeviceMaterial)*N);

    free(temp_h_m);
}

std::ostream &operator<<(std::ostream &os, const Material &m) {
    return os << "{rho: " << m.Rho() << ", P: ["
                        << m.Cp() << ", " << m.Qp() << "], S: ["
                        << m.Cs() << ", " << m.Qs() << "]}";
}

std::ostream &operator<<(std::ostream &os, const Material *m) {
    return os << "{rho: " << m->Rho() << ", P: ["
                        << m->Cp() << ", " << m->Qp() << "], S: ["
                        << m->Cs() << ", " << m->Qs() << "]}";
}