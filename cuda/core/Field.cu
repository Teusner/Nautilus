#include "Field.cuh"

PressureField::PressureField(dim3 d) : Field(d) {
    unsigned int size = m_dim.x * m_dim.y * m_dim.z;
    x = thrust::device_vector<float> (size, 0);
    y = thrust::device_vector<float> (size, 0);
    z = thrust::device_vector<float> (size, 0);
    xy = thrust::device_vector<float> (size, 0);
    yz = thrust::device_vector<float> (size, 0);
    xz = thrust::device_vector<float> (size, 0);
}

VelocityField::VelocityField(dim3 d) : Field(d) {
    unsigned int size = m_dim.x * m_dim.y * m_dim.z;
    x = thrust::device_vector<float> (size, 0);
    y = thrust::device_vector<float> (size, 0);
    z = thrust::device_vector<float> (size, 0);
}

MemoryField::MemoryField(dim3 d) : Field(d) {
    unsigned int size = m_dim.x * m_dim.y * m_dim.z;
    x = thrust::device_vector<float> (size, 0);
    y = thrust::device_vector<float> (size, 0);
    z = thrust::device_vector<float> (size, 0);
    xy = thrust::device_vector<float> (size, 0);
    yz = thrust::device_vector<float> (size, 0);
    xz = thrust::device_vector<float> (size, 0);
}