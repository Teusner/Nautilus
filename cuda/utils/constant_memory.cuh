#pragma once
#include <vector>
#include "core/Material.cuh"
#include "core/Module.cuh"

void CopyMaterialToSymbol(const void* symbol, const std::vector<Material> &materials);

struct CheckUndefinedMaterial {
    const unsigned int size;
    CheckUndefinedMaterial(unsigned int _size) : size(_size) {};
    __host__ __device__ bool operator()(const unsigned int &v) const {
        return (bool)(v >= size);
    }
};