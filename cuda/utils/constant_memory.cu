#include "constant_memory.cuh"
#include "core/Material.cuh"
#include <vector>
#include "core/FrequencyDomain.cuh"

void CopyMaterialToSymbol(const void* symbol, const std::vector<Material> &materials, unsigned int l, std::vector<float> tau_sigma, FrequencyDomain fd) {
    unsigned int n = materials.size();
    std::vector<DeviceMaterial> dm;
    for (auto const & m : materials) {
        dm.push_back(m.GetDeviceMaterial(l, tau_sigma, fd));
    }
    cudaMemcpyToSymbol(symbol, dm.data(), sizeof(DeviceMaterial)*n);
}
