#pragma once
#include <thrust/device_vector.h>


class Field {
    /// Default Constructor
    public: Field() : m_dim(0, 0, 0) {};

    /// Constructor with Field size
    public: Field(dim3 d) : m_dim(d) {};

    /// Constructor with Field size
    public: Field(const unsigned int x, const unsigned int y, const unsigned int z) : Field(dim3(x, y, z)) {};
    
    /// Size Getter
    public: dim3 Dim() const { return m_dim; };

    /// X Getter
    public: unsigned int X() const { return m_dim.x; };
    
    /// Y Getter
    public: unsigned int Y() const { return m_dim.y; };

    /// Z Getter
    public: unsigned int Z() const { return m_dim.z; };

    /// Field size
    protected: const dim3 m_dim;
};

class PressureField : public Field {
    /// Constructor
    public: PressureField(dim3 d);

    public: PressureField(const unsigned int x, const unsigned int y, const unsigned int z) : PressureField(dim3(x, y, z)) {};

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

    public: VelocityField(const unsigned int x, const unsigned int y, const unsigned int z) : VelocityField(dim3(x, y, z)) {}

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

    public: MemoryField(const unsigned int x, const unsigned int y, const unsigned int z) : MemoryField(dim3(x, y, z)) {}

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


/// PressureField Declaration
inline PressureField::PressureField(dim3 d) : Field(d) {
    unsigned int size = m_dim.x * m_dim.y * m_dim.z;
    x = thrust::device_vector<float>(size, 0);
    y = thrust::device_vector<float>(size, 0);
    z = thrust::device_vector<float>(size, 0);
    xy = thrust::device_vector<float>(size, 0);
    yz = thrust::device_vector<float>(size, 0);
    xz = thrust::device_vector<float>(size, 0);
}

/// VelocityField Declaration
inline VelocityField::VelocityField(dim3 d) : Field(d) {
    unsigned int size = m_dim.x * m_dim.y * m_dim.z;
    x = thrust::device_vector<float> (size, 0);
    y = thrust::device_vector<float> (size, 0);
    z = thrust::device_vector<float> (size, 0);
}

/// MemoryField Declaration
inline MemoryField::MemoryField(dim3 d) : Field(d) {
    unsigned int size = m_dim.x * m_dim.y * m_dim.z;
    x = thrust::device_vector<float> (size, 0);
    y = thrust::device_vector<float> (size, 0);
    z = thrust::device_vector<float> (size, 0);
    xy = thrust::device_vector<float> (size, 0);
    yz = thrust::device_vector<float> (size, 0);
    xz = thrust::device_vector<float> (size, 0);
}