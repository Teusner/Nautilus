#pragma once

#include <functional>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

class Module {
    public:
        __host__ Module(dim3 d) : m_x(d.x), m_y(d.y), m_z(d.z) {};
        __host__ Module(unsigned int i, unsigned int j, unsigned int k) : m_x(i), m_y(j), m_z(k) {};
        __host__ void print();

        __host__ unsigned int X() const { return m_x; };
        __host__ unsigned int Y() const { return m_y; };
        __host__ unsigned int Z() const { return m_z; };

    protected:
        /// X Position
        unsigned int m_x;

        /// Y Position
        unsigned int m_y;

        /// Z Position
        unsigned int m_z;

        thrust::host_vector<float> m_t;
        thrust::host_vector<float> m_s;
};

class Emitter : public Module {
    public:
        __host__ Emitter() : Module(0, 0, 0), m_f([](float t){return 0;}) {};
        __host__ Emitter(dim3 d, std::function<float(float)> f) : Module(d), m_f(f) {};
        __host__ Emitter(unsigned int i, unsigned int j, unsigned int k, std::function<float(float)> f) : Module(i, j, k), m_f(f) {};
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
