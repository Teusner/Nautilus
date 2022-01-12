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
    CUDA_CHECK( cudaMalloc(&this->m_device_ptr, sizeof(Material)) );
    this->update_device();
}

Material::~Material() {
    cudaFree(this->m_device_ptr);
}

void Material::update_host() {
    CUDA_CHECK( cudaMemcpy(this->host_ptr(), this->device_ptr(), sizeof(Material), cudaMemcpyDeviceToHost) );
}

void Material::update_device() {
    CUDA_CHECK( cudaMemcpy(this->device_ptr(), this->host_ptr(), sizeof(Material), cudaMemcpyHostToDevice) );
}