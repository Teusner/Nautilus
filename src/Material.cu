#include "Material.h"

#include <thrust/device_malloc.h>
#include <thrust/device_vector.h>

#include <stdio.h>

#include <sstream>
#define CUDA_CHECK( call )                                                 \
    do {                                                                   \
        cudaError_t code = call;                                           \
        if(code != cudaSuccess) {                                          \
            std::ostringstream oss;                                        \
            oss << "CUDA call '" << #call << "' failed '"                  \
                << cudaGetErrorString(code) << "' (code:" << code << ")\n" \
                << __FILE__ << ":" << __LINE__ << "\n";                    \
            throw std::runtime_error(oss.str());                           \
        }                                                                  \
    } while(0)                                                             \

#define CUDA_CHECK_LAST( call )                                            \
    do {                                                                   \
        cudaError_t code = cudaGetLastError();                             \
        if(code != cudaSuccess) {                                          \
            std::ostringstream oss;                                        \
            oss << "CUDA call '" << #call << "' failed '"                  \
                << cudaGetErrorString(code) << "' (code:" << code << ")\n" \
                << __FILE__ << ":" << __LINE__ << "\n";                    \
            throw std::runtime_error(oss.str());                           \
        }                                                                  \
    } while(0)                                                             \

#define N 10


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

Material::Material(float rho, float cp, float Qp) : m_rho(rho), m_cp(cp), m_Qp(Qp) {
}

Material::~Material() {
}

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
