#include "Constants.cuh"

#include "Scene.cuh"
#include "core/Material.cuh"
#include "utils/constant_memory.cuh"

#include <thrust/device_vector.h>

#include <stdexcept>

Scene::Scene(   const unsigned int x, const unsigned int y, const unsigned int z,
                const float dx, const float dy, const float dz, const float dt, FrequencyDomain frequency_domain
            ) : m_d(x, y, z), m_dx({dx, dy, dz}), m_dt(dt), m_frequency_domain(frequency_domain), P(x*y*z), U(x*y*z), R(x*y*z*frequency_domain.l())
{
    m_materials = std::vector<Material>(1, Material());
    m_M = thrust::device_vector<float>(x * y * z, 0);
    E = thrust::device_vector<float> (x * y * z, 0);
}

void Scene::AddMaterial(Material m) {
    m_materials.push_back(m);
}

void Scene::PrintMaterials() const {
    std::cout << "Materials : ";
    for (auto const &m : m_materials)
        std::cout << m << " ";
}

void Scene::SetScene(thrust::device_vector<unsigned int> &M) {
    // Checking scene size
    if (M.size() != m_d.x * m_d.y * m_d.z)
        throw std::invalid_argument("Scene vector size does not match the Scene dimensions !");

    // Checking undefined material
    if (thrust::transform_reduce(M.begin(), M.end(), CheckUndefinedMaterial(m_materials.size()), false, thrust::plus<bool>()))
        throw std::invalid_argument("Scene vector contains uninitialized materials !");

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

void Scene::Init() const {
    /// Allocating Time Step
    cudaMemcpyToSymbol(d_dt, &m_dt, sizeof(float), 0, cudaMemcpyHostToDevice);

    /// Allocating alpha coefficients
    std::vector<float> alpha = {1.f / (24.f*m_dx.x), 1.f / (24.f*m_dx.y), 1.f / (24.f*m_dx.z)};
    cudaMemcpyToSymbol(d_alpha, alpha.data(), sizeof(float)*3, 0, cudaMemcpyHostToDevice);

    /// Allocating Relaxation Constraints Number
    unsigned int l = m_frequency_domain.l();
    cudaMemcpyToSymbol(d_l, &l, sizeof(unsigned int), 0, cudaMemcpyHostToDevice);

    /// Allocating DeviceMaterial
    unsigned int n = m_materials.size();
    std::vector<DeviceMaterial> dm;
    std::transform(std::begin(m_materials), std::end(m_materials), std::back_inserter(dm), [&] (Material m) { return m.GetDeviceMaterial(m_frequency_domain); });
    cudaMemcpyToSymbol(M, dm.data(), sizeof(DeviceMaterial)*n);

    /// Allocating Tau Sigma
    std::vector<float> tau_sigma = m_frequency_domain.TauSigma();
    cudaMemcpyToSymbol(d_tau_sigma, tau_sigma.data(), sizeof(float)*l, 0, cudaMemcpyHostToDevice);
}