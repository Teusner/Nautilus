#include "Scene.h"
#include "core/Material.cuh"
#include "utils/constant_memory.cuh"


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