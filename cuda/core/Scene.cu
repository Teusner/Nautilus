#include "Scene.cuh"
#include "FrequencyDomain.cuh"
#include "core/Material.cuh"
#include "utils/constant_memory.cuh"

#include <thrust/device_vector.h>

#include <stdexcept>


Scene::Scene(   const unsigned int x, const unsigned int y, const unsigned int z,
                const float dx, const float dy, const float dz, const float dt, FrequencyDomain frequency_domain
            ) : m_d(x, y, z), m_dx({dx, dy, dz}), m_dt(dt), m_alpha(3), m_device_materials(0), m_frequency_domain(frequency_domain), P(x*y*z), U(x*y*z), dU(x*y*z), R(x*y*z*frequency_domain.l())
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

void Scene::Init() {
    m_alpha[0] = 1.f / (24.f * m_dx.x);
    m_alpha[1] = 1.f / (24.f * m_dx.y);
    m_alpha[2] = 1.f / (24.f * m_dx.z);

    /// Allocating DeviceMaterial
    FrequencyDomain fd = m_frequency_domain;
    for (const auto &m : m_materials) {
        m_device_materials.push_back(m.GetDeviceMaterial<float>(fd));
    }

    // std::cout << "Eta Tau P : [";
    // thrust::copy(m_device_materials.eta_tau_p_1.begin(), m_device_materials.eta_tau_p_1.end(), std::ostream_iterator<float>(std::cout, " "));
    // std::cout << "]\n";
    // std::cout << "Mu Tau S : [";
    // thrust::copy(m_device_materials.mu_tau_s_1.begin(), m_device_materials.mu_tau_s_1.end(), std::ostream_iterator<float>(std::cout, " "));
    // std::cout << "]\n";

    /// Allocating Tau Sigma
    std::vector<float> tau_sigma = m_frequency_domain.TauSigma();
    // m_tau_sigma.resize(tau_sigma.size());
    // thrust::copy(m_tau_sigma.begin(), m_tau_sigma.end(), tau_sigma.begin());
    m_tau_sigma = thrust::device_vector<float>(tau_sigma.begin(), tau_sigma.end());
}