#pragma once

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "Material.h"

class Scene {
    public:
        __host__ Scene(dim3 d) : m_d(d) {};

        __host__ void AddMaterial(Material m);
        __host__ void PrintMaterials();

    private:
        /// Scene dimension
        dim3 m_d;

        /// Spatial step
        float dx;
        float dy;
        float dz;

        thrust::host_vector<Material> m_materials;
        thrust::host_vector<thrust::device_ptr<Material>> m_scene;
};