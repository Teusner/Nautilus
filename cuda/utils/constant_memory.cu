#include "constant_memory.cuh"
#include "core/Material.cuh"
#include <vector>

void CopyMaterialToSymbol(const void* symbol, const std::vector<Material> &materials) {
    unsigned int n = materials.size();
    std::vector<DeviceMaterial> dm;
    for (auto const & m : materials) {
        dm.push_back(m.GetDeviceMaterial());
    }
    cudaMemcpyToSymbol(symbol, dm.data(), sizeof(DeviceMaterial)*n);
}
