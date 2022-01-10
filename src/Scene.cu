#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>


#include "Material.h"
#include "Scene.h"

void Scene::AddMaterial(thrust::device_ptr<Material> m) {
    d_materials.push_back(m);
}

void Scene::PrintMaterials() {
    // h_materials = d_materials;
    std::cout << "Materials : ";
    for (auto const &m : d_materials)
        std::cout << m << " ";
}