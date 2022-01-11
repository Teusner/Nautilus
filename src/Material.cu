#include "Material.h"

#include <thrust/device_malloc.h>
#include <thrust/device_vector.h>

#include <stdio.h>

std::ostream &operator<<(std::ostream &os, const Material& m) {
    return os << "{rho: " << m.Rho() << ", P: ["
                        << m.Cp() << ", " << m.Qp() << "], S: ["
                        << m.Cs() << ", " << m.Qs() << "]}";
}

Material::Material(float rho, float cp, float Qp) : m_rho(rho), m_cp(cp), m_Qp(Qp) {
    cudaMalloc(&this->m_device_ptr, sizeof(Material));
}

Material::~Material() {
    cudaFree(&this->m_device_ptr);
}

void Material::update_host() {
    cudaMemcpy(this->host_ptr(), this->device_ptr(), sizeof(Material), cudaMemcpyDeviceToHost);
}

void Material::update_device() {
    cudaMemcpy(this->device_ptr(), this->host_ptr(), sizeof(Material), cudaMemcpyHostToDevice);
}