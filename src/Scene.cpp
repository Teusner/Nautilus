#include "Material.h"
#include "Scene.h"

void Scene::AddMaterial(Material m) {
    m_materials.push_back(m);
}

void Scene::PrintMaterials() {
    std::cout << "Materials : ";
    for (auto const &m : m_materials)
        std::cout << m << " ";
}