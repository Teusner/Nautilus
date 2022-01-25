#pragma once

#include <iostream>
#include <string>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

struct DeviceEmitter {
    unsigned int x, y, z;
    float (*f)(float);
};

class Module {
    public:
        /// Default constructor
        __host__ Module() {};

        /// Constructor with grid position as unsigned int
        __host__ Module(unsigned int i, unsigned int j, unsigned int k) : x(i), y(j), z(k) {};

        /// Constructor with grid position as dim3
        __host__ Module(dim3 d) : Module(d.x, d.y, d.z) {};

        /// String representation
        __host__ virtual std::string String() const { return std::string("Module"); };

        /// X Position
        unsigned int x;

        /// Y Position
        unsigned int y;

        /// Z Position
        unsigned int z;
};

class Emitter : public Module {
    public:
        /// Default Constructor
        __host__ Emitter() {};

        /// Constructor with grid position as unsigned int and function
        __host__ Emitter(unsigned int i, unsigned int j, unsigned int k, float (*f)(float)) : Module(i, j, k), m_f(f) {};

        /// Constructor with grid position as dim3 and function
        __host__ Emitter(dim3 d, float (*f)(float)) : Module(d), m_f(f) {};

        /// Call operator to get produced signal
        __host__ float operator()(float x) const { return m_f(x); };

        /// String representation of the Emitter
        __host__ std::string String() const override { return std::string("Emitter"); };

        __host__ DeviceEmitter GetDeviceEmitter() const { return (DeviceEmitter){x, y, z, m_f}; };

    private:
        float (*m_f)(float);
        // std::function<float(float)> m_f;
};

class Reciever : public Module {
    public:
        /// Default Constructor
        __host__ Reciever() {};

        /// Constructor with position as unsigned int
        __host__ Reciever(unsigned int i, unsigned int j, unsigned int k) : Module(i, j, k) {};

        /// Constructor with position as dim3
        __host__ Reciever(dim3 d) : Module(d) {};

        /// Record a signal s recieved at time t
        __host__ void Record(float t, float s);

        /// String representation of the Reciever
        __host__ std::string String() const override { return std::string("Reciever"); };

        // Time Getter
        __host__ thrust::host_vector<float> T() const { return m_t; };

        /// Signal Getter
        __host__ thrust::host_vector<float> S() const { return m_s; };

    private:
        thrust::host_vector<float> m_t;
        thrust::host_vector<float> m_s;
};

// Implementation
inline std::ostream& operator<<(std::ostream& os, const Module &m) {
    return os << m.String() << " : {" << m.x << ", " << m.y << ", " << m.z <<  "}\n" ;
}

inline void Reciever::Record(float t, float s) {
    m_t.push_back(t);
    m_s.push_back(s);
}