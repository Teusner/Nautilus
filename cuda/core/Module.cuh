#pragma once

#include <iostream>
#include <string>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>


class Module {
    public:
        /// Default constructor
        __host__ __device__ Module() {};

        /// Constructor with grid position as unsigned int
        __host__ __device__ Module(unsigned int i, unsigned int j, unsigned int k) : x(i), y(j), z(k) {};

        /// Constructor with grid position as dim3
        __host__ __device__ Module(dim3 d) : Module(d.x, d.y, d.z) {};

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
        __host__ __device__ Emitter() {};

        /// Constructor with grid position as unsigned int and function
        __host__ __device__ Emitter(unsigned int i, unsigned int j, unsigned int k, float (*f)(float)) : Module(i, j, k), m_f(f) {};

        /// Constructor with grid position as dim3 and function
        __host__ __device__ Emitter(dim3 d, float (*f)(float)) : Module(d), m_f(f) {};

        /// Call operator to get produced signal
        __host__ __device__ float operator()(float x) const { return m_f(x); };

        /// String representation of the Emitter
        __host__ std::string String() const override { return std::string("Emitter"); };

    private:
        float (*m_f)(float);
};

class SinEmitter : public Module {
    // Default Constructor
    public:  __host__ __device__ SinEmitter() {};

    /// Constructor with grid position as unsigned int and function
    public: __host__ __device__ SinEmitter(unsigned int i, unsigned int j, unsigned int k) : Module(i, j, k) {};

    /// Constructor with grid position as dim3 and function
    public: __host__ __device__ SinEmitter(dim3 d) : Module(d) {};

    /// Call operator
    public: __host__ __device__ float operator()(float t) const { return 0.1*sin(2*M_PI*10*t); };
};

class Reciever : public Module {
    public:
        /// Default Constructor
        __host__ __device__ Reciever() {};

        /// Constructor with position as unsigned int
        __host__ __device__ Reciever(unsigned int i, unsigned int j, unsigned int k) : Module(i, j, k) {};

        /// Constructor with position as dim3
        __host__ __device__ Reciever(dim3 d) : Module(d) {};

        /// Record a signal s recieved at time t
        __host__ __device__ void Record(float t, float s);

        /// String representation of the Reciever
        __host__ std::string String() const override { return std::string("Reciever"); };

        // Time Getter
        __host__ __device__ thrust::host_vector<float> T() const { return m_t; };

        /// Signal Getter
        __host__ __device__ thrust::host_vector<float> S() const { return m_s; };

    private:
        thrust::host_vector<float> m_t;
        thrust::host_vector<float> m_s;
};

// Implementation
inline std::ostream& operator<<(std::ostream& os, const Module &m) {
    return os << m.String() << " : {" << m.x << ", " << m.y << ", " << m.z <<  "}\n" ;
}

inline __host__ __device__ void Reciever::Record(float t, float s) {
    // m_t.push_back(t);
    // m_s.push_back(s);
}