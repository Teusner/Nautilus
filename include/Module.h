#pragma once

#include <functional>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

class Module {
    public:
        __host__ Module(dim3 d) : m_d(d) {};
        __host__ Module(std::size_t i, std::size_t j, std::size_t k) : m_d(i, j, k) {};
        __host__ dim3 Dim() const {return m_d;};
        __host__ void print();

    protected:
        dim3 m_d;
        thrust::host_vector<float> m_t;
        thrust::host_vector<float> m_s;
};

class Emitter : public Module {
    public:
        __host__ Emitter(dim3 d, std::function<float(float)> f) : Module(d), m_f(f) {};
        __host__ Emitter(std::size_t i, std::size_t j, std::size_t k, std::function<float(float)> f) : Module(i, j, k), m_f(f) {};
        __host__ float operator()(float x);

    private:
        std::function<float(float)> m_f;
};

class Reciever : public Module {
    public:
        __host__ Reciever(dim3 d) : Module(d) {};
        __host__ Reciever(std::size_t i, std::size_t j, std::size_t k) : Module(i, j, k) {};

        __host__ void Record(float t, float s);
};
