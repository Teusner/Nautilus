#include "Scene.h"
#include "core/Material.cuh"
#include "utils/constant_memory.cuh"

#include <memory>
#include <stdexcept>

Scene::Scene(dim3 d) {
    m_d = d;
    m_materials = std::vector<Material>(1, Material());
    m_M = std::vector<float>(d.x * d.y * d.z, 0);

    P = std::make_unique<PressureField>(d);
    U = std::make_unique<VelocityField>(d);
    R = std::make_unique<MemoryField>(d);
}

void Scene::AddMaterial(Material m) {
    m_materials.push_back(m);
}

void Scene::PrintMaterials() const {
    std::cout << "Materials : ";
    for (auto const &m : m_materials)
        std::cout << m << " ";
}

void Scene::AllocateMaterials(const void* symbol) const {
    CopyMaterialToSymbol(symbol, m_materials);
}

void Scene::SetScene(std::vector<float> M) {
    if (M.size() != m_d.x * m_d.y * m_d.z) {
        throw std::invalid_argument("Scene vector size does not match the Scene dimensions !");
    }

    unsigned int s = m_materials.size();
    for (const auto & v : M) {
        if (v >= s) {
            throw std::invalid_argument("Scene vector contains uninitialized materials !");
        }
    }

    m_M = M;
}