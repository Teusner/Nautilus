#include "Scene.cuh"
#include "core/Material.cuh"
#include "utils/constant_memory.cuh"

#include <thrust/device_vector.h>

#include <stdexcept>

Scene::Scene(dim3 d, float dx, float dy, float dz) : m_d(d), m_dx(dx), m_dy(dy), m_dz(dz), P(d), U(d), R(d){
    m_materials = std::vector<Material>(1, Material());
    m_M = std::vector<float>(d.x * d.y * d.z, 0);
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

void Scene::SetScene(thrust::device_vector<float> M) {
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

void Scene::TriggerNextEvent() {
    Event e = m_events.top();
    if (m_i < e.i())
        throw std::invalid_argument("Scene time is prior to the time of the next Event !");

    if (m_i > e.i())
        throw std::invalid_argument("Scene time is later to the time of the next Event !");

    e.Callback();
    m_events.pop();
}