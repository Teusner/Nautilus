#pragma once

#include <thrust/device_ptr.h>
#include <thrust/detail/config/host_device.h>
#include <iostream>


class Material {
    public:
        __host__ Material() : Material(1000, 1500, 100) {};
        __host__ Material(float rho, float cp, float Qp);
        __host__ ~Material();

        __device__ __host__ float Rho() const { return m_rho; };
        __device__ __host__ float Cp() const { return m_cp; };
        __device__ __host__ float Qp() const { return m_Qp; };
        __device__ __host__ float Cs() const { return m_cs; };
        __device__ __host__ float Qs() const { return m_Qs; };

        __device__ __host__ Material* host_ptr() { return this; }
        __device__ __host__ const Material* host_ptr() const { return this; }
        __device__ __host__ Material* device_ptr() const { return m_device_ptr; };

        __host__ void update_host();
        __host__ void update_device();

    private:
        float m_rho;
        float m_cp;
        float m_Qp;
        float m_cs = 0;
        float m_Qs = 0;

        Material* m_device_ptr;
};


std::ostream &operator<<(std::ostream &os, const Material& m);

__global__ void NewMaterial(float rho, float cp, float qp, Material* m);

__global__ void PrintMaterial(Material* m);