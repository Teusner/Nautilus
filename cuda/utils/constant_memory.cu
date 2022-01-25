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

void CopyEmitterToSymbol(const void* symbol, const std::vector<Emitter> &emitters) {
    unsigned int n = emitters.size();
    std::vector<DeviceEmitter> de;
    for (auto const & e : emitters) {
        de.push_back(e.GetDeviceEmitter());
    }
    cudaMemcpyToSymbol(symbol, de.data(), sizeof(DeviceEmitter)*n);
}