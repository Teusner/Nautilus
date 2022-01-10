#pragma once

#include <thrust/detail/config/host_device.h>

#include <iostream>


class Material {
    public:
        __device__ __host__ Material(float rho, float cp, float Qp) : m_rho(rho), m_cp(cp), m_Qp(Qp) {};

        __host__ float Rho() const {return m_rho;};
        __host__ float Cp() const {return m_cp;};
        __host__ float Qp() const {return m_Qp;};
        __host__ float Cs() const {return m_cs;};
        __host__ float Qs() const {return m_Qs;};

    private:
        float m_rho;
        float m_cp;
        float m_Qp;
        float m_cs = 0;
        float m_Qs = 0;
};


std::ostream &operator<<(std::ostream &os, const Material& m);