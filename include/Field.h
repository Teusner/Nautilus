#pragma once

#include <thrust/device_vector.h>


class Field {
    /// Default Constructor
    public: Field() : m_dim(0, 0, 0) {};

    /// Constructor with Field size
    public: Field(dim3 d) : m_dim(d) {};
    
    /// Size Getter
    public: dim3 Dim() const { return m_dim; };

    /// Field size
    protected: dim3 m_dim;
};

class PressureField : public Field {
    /// Constructor
    public: PressureField(dim3 d);

    /// X Pressure
    public: thrust::device_vector<float> x;

    /// Y Pressure
    public: thrust::device_vector<float> y;

    /// Z Pressure
    public: thrust::device_vector<float> z;

    /// XY Pressure
    public: thrust::device_vector<float> xy;

    /// Yz Pressure
    public: thrust::device_vector<float> yz;

    /// XZ Pressure
    public: thrust::device_vector<float> xz;
};

class VelocityField : public Field {

    /// Constructor
    public: VelocityField(dim3 d);

    /// X Velocity
    public: thrust::device_vector<float> x;

    /// Y Velocity
    public: thrust::device_vector<float> y;

    /// Z Velocity
    public: thrust::device_vector<float> z;
};

class MemoryField : public Field {

    /// Constructor
    public: MemoryField(dim3 d);

    /// X Memory
    public: thrust::device_vector<float> x;

    /// Y Memory
    public: thrust::device_vector<float> y;

    /// Z Memory
    public: thrust::device_vector<float> z;

    /// XY Memory
    public: thrust::device_vector<float> xy;

    /// YZ Memory
    public: thrust::device_vector<float> yz;

    /// XZ Memory
    public: thrust::device_vector<float> xz;
};