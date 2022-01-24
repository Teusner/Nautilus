#include "Scene.cuh"
#include "core/Material.cuh"
#include "utils/constant_memory.cuh"

#include <thrust/device_vector.h>

#include <stdexcept>

Scene::Scene(   const unsigned int x, const unsigned int y, const unsigned int z,
                const float dx, const float dy, const float dz, const float dt
            ) : m_d(x, y, z), m_dx({dx, dy, dz}), m_dt(dt), P(x, y, z), U(x, y, z), R(x, y, z)
{
    m_materials = std::vector<Material>(1, Material());
    m_M = std::vector<float>(x * y * z, 0);
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