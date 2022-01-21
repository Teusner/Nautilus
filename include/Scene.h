#pragma once
#include <memory>
#include <vector>
#include "core/Material.cuh"
#include "core/Field.cuh"


class Scene {
    public:

        /// Constructor with a dimension
        Scene(dim3 d);

        /// Copy Constructor
        Scene(const Scene& s) : m_d(s.Dims()) {};

        /// Move constructor
        Scene(Scene&&) = default;

        /// Scene Dimension Getter
        dim3 Dims() const { return m_d; };

        /// Add a material to the Scene
        void AddMaterial(Material m);

        /// Print material in the scene
        void PrintMaterials() const;

        /// Copy materials to constant memory
        void AllocateMaterials(const void* symbol) const;

        /// Scene Matrix Getter
        std::vector<float> GetScene() const { return m_M; };

        /// Scene Matrix Setter
        void SetScene(std::vector<float> M);

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

        /// Pressure Field
        std::unique_ptr<PressureField> P;

        /// Velocity Field
        std::unique_ptr<VelocityField> U;

        /// Memory Field
        std::unique_ptr<MemoryField> R;
};