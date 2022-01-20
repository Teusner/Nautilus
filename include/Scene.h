#pragma once
#include <vector>
#include "core/Material.cuh"


class Scene {
    public:
        Scene(dim3 d) : m_d(d), m_materials(1, Material()) {};

        /// Add a material to the Scene
        void AddMaterial(Material m);

        /// Print material in the scene
        void PrintMaterials() const;

        /// Copy materials to constant memory
        void AllocateMaterials(const void* symbol) const;

    private:
        /// Scene dimension
        dim3 m_d;

        /// Spatial step
        float dx;
        float dy;
        float dz;

        /// Vector of Material
        std::vector<Material> m_materials;

        /// Scene description vector
        std::vector<float> m_M;
};