#pragma once

#include <queue>
#include <vector>

#include "Material.cuh"
#include "Field.cuh"
#include "Event.h"
#include "Module.cuh"
#include "FrequencyDomain.cuh"
#include "kernels.cuh"
#include "utils/constant_memory.cuh"


template <unsigned int x, unsigned int y, unsigned int z, unsigned int N, typename T>
class Scene {
    /// Constructor with a dimension
    public: Scene(const float dx, const float dy, const float dz, const float dt, FrequencyDomain frequency_domain);


    /// ### Scene Dimensions
    /// Scene Dimension Getter
    public: dim3 Dims() const { return m_d; };

    /// X Getter
    public: unsigned int X() const { return m_d.x; };

    /// Y Getter
    public: unsigned int Y() const { return m_d.y; };

    /// Z Getter
    public: unsigned int Z() const { return m_d.z; };

    /// X-Step Getter
    public: float dX() const { return m_dx.x; };

    /// Y-Step Getter
    public: float dY() const { return m_dx.y; };

    /// Z-Step Getter
    public: float dZ() const { return m_dx.z; };

    /// Scene dimension
    private: const dim3 m_d;

    /// Spatial step
    private: const float3 m_dx;

    /// m_alpha = 1/(24*m_dx)
    private: thrust::device_vector<float> m_alpha;


    /// ### Material and Scene
    /// Add a material to the Scene
    public: void AddMaterial(Material m);

    /// Print material in the scene
    public: void PrintMaterials() const;

    /// Scene Matrix Getter
    public: thrust::device_vector<unsigned int> GetScene() const { return m_M; };

    /// Scene Matrix Setter
    public: void SetScene(thrust::device_vector<unsigned int> &M);

    /// Vector of Material
    private: std::vector<Material> m_materials;

    /// Device Allocated Materials (SOA)
    private: DeviceMaterials<thrust::device_vector<float>> m_device_materials;

    /// Scene description vector
    private: thrust::device_vector<unsigned int> m_M;



    /// ### Simulation Time
    /// Return the current simulation time
    public: double Time() const { return double(m_dt * m_i); };

    /// Time Step Getter
    public: float TimeStep() const { return m_dt; };

    /// Time Increment
    /// Return the current time increment
    public: unsigned int Increment() const { return m_i; };

    /// Time Step
    private: float m_dt;
    
    /// Time Increment
    public: unsigned int m_i;



    /// ### Emitters and Recievers
    /// Vector of Emmitters
    public: thrust::device_vector<SinEmitter> emitters;


    
    /// ### Fields
    /// Pressure Field
    public: PressureField<thrust::device_vector<float>> P;

    /// Velocity Field
    public: VelocityField<thrust::device_vector<float>> U;

    /// Memory Field
    public: MemoryField<thrust::device_vector<float>> R;

    /// Derivative of Velocity Field
    private: VelocityField<thrust::device_vector<float>> dU;

    /// Emitter Pressure Field
    private: thrust::device_vector<float> E;

    /// Absorbing Boundary Field
    private: thrust::device_vector<float> B;



    /// ### Frequency Domain
    /// Number of relaxation constraints getter
    public: unsigned int l() const { return m_frequency_domain.l(); };

    /// FrequencyDomain Getter
    public: FrequencyDomain GetFrequencyDomain() const { return m_frequency_domain; };

    /// FrequencyDomain
    private: FrequencyDomain m_frequency_domain;

    private: thrust::device_vector<float> m_tau_sigma;



    /// ### Event Based Simulation
    /// Next Event Getter
    public: Event GetNextEvent() const { return m_events.top(); };

    /// Next Event Trigger
    public: void TriggerNextEvent();

    /// Initialize the Scene
    public: void Init();

    /// Step
    public: void Step();

    /// Priority queue handling Events in priority order
    private : std::priority_queue<Event, std::vector<Event>, std::greater<Event>> m_events;
};


/// Implementation
template <unsigned int x, unsigned int y, unsigned int z, unsigned int N, typename T>
Scene<x, y, z, N, T>::Scene(const float dx, const float dy, const float dz, const float dt, FrequencyDomain frequency_domain) : m_d(x, y, z), m_dx({dx, dy, dz}), m_dt(dt), m_alpha(3), m_device_materials(0), m_frequency_domain(frequency_domain), P(x*y*z), U(x*y*z), dU(x*y*z), B(x*y*z), R(x*y*z*frequency_domain.l()) {
    m_materials = std::vector<Material>(1, Material());
    m_M = thrust::device_vector<float>(x * y * z, 0);
    E = thrust::device_vector<float> (x * y * z, 0);

    /// Absorbing Boundary Condition
    float zeta_min = 0.95;
    float p = 1.6;

    auto Op = Bound<x, y, z, N>(zeta_min, p);
    thrust::counting_iterator<int> idxfirst(0);
    thrust::counting_iterator<int> idxlast = idxfirst + m_d.x * m_d.y * m_d.z;
    thrust::transform(idxfirst, idxlast, B.begin(), B.begin(), Op);
}

template<unsigned int x, unsigned int y, unsigned int z, unsigned int N, typename T>
void Scene<x, y, z, N, T>::Step() {
    // Emitter Field computing
    F<x, y, z, T><<<1, 1>>>(m_dt*m_i, thrust::raw_pointer_cast(&(emitters[0])), thrust::raw_pointer_cast(&(E[0])));

    dim3 ThreadPerBlock(4, 4, 4);
    dim3 GridDimension(x / ThreadPerBlock.x, y / ThreadPerBlock.y, z / ThreadPerBlock.z);

    Ux<x, y, z><<<GridDimension, ThreadPerBlock>>>(
        m_dt,
        thrust::raw_pointer_cast(&(m_alpha[0])),
        thrust::raw_pointer_cast(&(U.x[0])),
        thrust::raw_pointer_cast(&(P.x[0])),
        thrust::raw_pointer_cast(&(P.xy[0])),
        thrust::raw_pointer_cast(&(P.xz[0])),
        thrust::raw_pointer_cast(&(m_M[0])),
        thrust::raw_pointer_cast(&(m_device_materials.inv_rho[0]))
    );

    Uy<x, y, z><<<GridDimension, ThreadPerBlock>>>(
        m_dt,
        thrust::raw_pointer_cast(&(m_alpha[0])),
        thrust::raw_pointer_cast(&(U.y[0])),
        thrust::raw_pointer_cast(&(P.y[0])),
        thrust::raw_pointer_cast(&(P.xy[0])),
        thrust::raw_pointer_cast(&(P.yz[0])),
        thrust::raw_pointer_cast(&(m_M[0])),
        thrust::raw_pointer_cast(&(m_device_materials.inv_rho[0]))
    );

    Uz<x, y, z><<<GridDimension, ThreadPerBlock>>>(
        m_dt,
        thrust::raw_pointer_cast(&(m_alpha[0])),
        thrust::raw_pointer_cast(&(U.z[0])),
        thrust::raw_pointer_cast(&(P.z[0])),
        thrust::raw_pointer_cast(&(P.yz[0])),
        thrust::raw_pointer_cast(&(P.xz[0])),
        thrust::raw_pointer_cast(&(m_M[0])),
        thrust::raw_pointer_cast(&(m_device_materials.inv_rho[0]))
    );

    // Let each kernels finising their tasks
    // cudaDeviceSynchronize();

    Uxx<x, y, z><<<GridDimension, ThreadPerBlock>>>(
        m_alpha[0],
        thrust::raw_pointer_cast(&(U.x[0])),
        thrust::raw_pointer_cast(&(dU.x[0]))
    );

    Uyy<x, y, z><<<GridDimension, ThreadPerBlock>>>(
        m_alpha[1],
        thrust::raw_pointer_cast(&(U.y[0])),
        thrust::raw_pointer_cast(&(dU.y[0]))
    );

    Uzz<x, y, z><<<GridDimension, ThreadPerBlock>>>(
        m_alpha[2],
        thrust::raw_pointer_cast(&(U.z[0])),
        thrust::raw_pointer_cast(&(dU.z[0]))
    );

    // Let each kernels finising their tasks
    // cudaDeviceSynchronize();

    Pxx<x, y, z><<<GridDimension, ThreadPerBlock>>>(
        m_dt,
        thrust::raw_pointer_cast(&(P.x[0])),
        thrust::raw_pointer_cast(&(dU.x[0])),
        thrust::raw_pointer_cast(&(dU.y[0])),
        thrust::raw_pointer_cast(&(dU.z[0])),
        thrust::raw_pointer_cast(&(R.x[0])),
        thrust::raw_pointer_cast(&(m_M[0])),
        thrust::raw_pointer_cast(&(m_device_materials.eta_tau_p[0])),
        thrust::raw_pointer_cast(&(m_device_materials.mu_tau_s[0])),
        thrust::raw_pointer_cast(&(E[0]))
    );

    Pyy<x, y, z><<<GridDimension, ThreadPerBlock>>>(
        m_dt,
        thrust::raw_pointer_cast(&(P.y[0])),
        thrust::raw_pointer_cast(&(dU.x[0])),
        thrust::raw_pointer_cast(&(dU.y[0])),
        thrust::raw_pointer_cast(&(dU.z[0])),
        thrust::raw_pointer_cast(&(R.y[0])),
        thrust::raw_pointer_cast(&(m_M[0])),
        thrust::raw_pointer_cast(&(m_device_materials.eta_tau_p[0])),
        thrust::raw_pointer_cast(&(m_device_materials.mu_tau_s[0])),
        thrust::raw_pointer_cast(&(E[0]))
    );

    Pzz<x, y, z><<<GridDimension, ThreadPerBlock>>>(
        m_dt,
        thrust::raw_pointer_cast(&(P.z[0])),
        thrust::raw_pointer_cast(&(dU.x[0])),
        thrust::raw_pointer_cast(&(dU.y[0])),
        thrust::raw_pointer_cast(&(dU.z[0])),
        thrust::raw_pointer_cast(&(R.z[0])),
        thrust::raw_pointer_cast(&(m_M[0])),
        thrust::raw_pointer_cast(&(m_device_materials.eta_tau_p[0])),
        thrust::raw_pointer_cast(&(m_device_materials.mu_tau_s[0])),
        thrust::raw_pointer_cast(&(E[0]))
    );

    Pxy<x, y, z><<<GridDimension, ThreadPerBlock>>>(
        m_dt,
        thrust::raw_pointer_cast(&(P.xy[0])),
        thrust::raw_pointer_cast(&(U.x[0])),
        thrust::raw_pointer_cast(&(U.y[0])),
        thrust::raw_pointer_cast(&(R.xy[0])),
        thrust::raw_pointer_cast(&(m_M[0])),
        thrust::raw_pointer_cast(&(m_device_materials.mu_tau_s[0]))
    );

    Pyz<x, y, z><<<GridDimension, ThreadPerBlock>>>(
        m_dt,
        thrust::raw_pointer_cast(&(P.yz[0])),
        thrust::raw_pointer_cast(&(U.y[0])),
        thrust::raw_pointer_cast(&(U.z[0])),
        thrust::raw_pointer_cast(&(R.yz[0])),
        thrust::raw_pointer_cast(&(m_M[0])),
        thrust::raw_pointer_cast(&(m_device_materials.mu_tau_s[0]))
    );

    Pxz<x, y, z><<<GridDimension, ThreadPerBlock>>>(
        m_dt,
        thrust::raw_pointer_cast(&(P.xz[0])),
        thrust::raw_pointer_cast(&(U.x[0])),
        thrust::raw_pointer_cast(&(U.z[0])),
        thrust::raw_pointer_cast(&(R.xz[0])),
        thrust::raw_pointer_cast(&(m_M[0])),
        thrust::raw_pointer_cast(&(m_device_materials.mu_tau_s[0]))
    );

    // Let each kernels finising their tasks
    // cudaDeviceSynchronize();

    Rxx<x, y, z><<<GridDimension, ThreadPerBlock>>>(
        m_dt,
        thrust::raw_pointer_cast(&(R.x[0])),
        thrust::raw_pointer_cast(&(dU.x[0])),
        thrust::raw_pointer_cast(&(dU.y[0])),
        thrust::raw_pointer_cast(&(dU.z[0])),
        thrust::raw_pointer_cast(&(m_M[0])),
        thrust::raw_pointer_cast(&(m_device_materials.eta_tau_p[0])),
        thrust::raw_pointer_cast(&(m_device_materials.mu_tau_s[0])),
        thrust::raw_pointer_cast(&(m_tau_sigma[0]))
    );

    Ryy<x, y, z><<<GridDimension, ThreadPerBlock>>>(
        m_dt,
        thrust::raw_pointer_cast(&(R.y[0])),
        thrust::raw_pointer_cast(&(dU.x[0])),
        thrust::raw_pointer_cast(&(dU.y[0])),
        thrust::raw_pointer_cast(&(dU.z[0])),
        thrust::raw_pointer_cast(&(m_M[0])),
        thrust::raw_pointer_cast(&(m_device_materials.eta_tau_p[0])),
        thrust::raw_pointer_cast(&(m_device_materials.mu_tau_s[0])),
        thrust::raw_pointer_cast(&(m_tau_sigma[0]))
    );

    Rzz<x, y, z><<<GridDimension, ThreadPerBlock>>>(
        m_dt,
        thrust::raw_pointer_cast(&(R.z[0])),
        thrust::raw_pointer_cast(&(dU.x[0])),
        thrust::raw_pointer_cast(&(dU.y[0])),
        thrust::raw_pointer_cast(&(dU.z[0])),
        thrust::raw_pointer_cast(&(m_M[0])),
        thrust::raw_pointer_cast(&(m_device_materials.eta_tau_p[0])),
        thrust::raw_pointer_cast(&(m_device_materials.mu_tau_s[0])),
        thrust::raw_pointer_cast(&(m_tau_sigma[0]))
    );

    Rxy<x, y, z><<<GridDimension, ThreadPerBlock>>>(
        m_dt,
        thrust::raw_pointer_cast(&(R.xy[0])),
        thrust::raw_pointer_cast(&(U.x[0])),
        thrust::raw_pointer_cast(&(U.y[0])),
        thrust::raw_pointer_cast(&(m_M[0])),
        thrust::raw_pointer_cast(&(m_device_materials.mu_tau_s[0])),
        thrust::raw_pointer_cast(&(m_tau_sigma[0]))
    );

    Ryz<x, y, z><<<GridDimension, ThreadPerBlock>>>(
        m_dt,
        thrust::raw_pointer_cast(&(R.yz[0])),
        thrust::raw_pointer_cast(&(U.y[0])),
        thrust::raw_pointer_cast(&(U.z[0])),
        thrust::raw_pointer_cast(&(m_M[0])),
        thrust::raw_pointer_cast(&(m_device_materials.mu_tau_s[0])),
        thrust::raw_pointer_cast(&(m_tau_sigma[0]))
    );

    Rxz<x, y, z><<<GridDimension, ThreadPerBlock>>>(
        m_dt,
        thrust::raw_pointer_cast(&(R.xz[0])),
        thrust::raw_pointer_cast(&(U.x[0])),
        thrust::raw_pointer_cast(&(U.z[0])),
        thrust::raw_pointer_cast(&(m_M[0])),
        thrust::raw_pointer_cast(&(m_device_materials.mu_tau_s[0])),
        thrust::raw_pointer_cast(&(m_tau_sigma[0]))
    );

    // Let each kernels finising their tasks
    // cudaDeviceSynchronize();

    // Finishing P update
    // auto func = saxpy_functor(- m_dt * 0.5);
    // thrust::transform(R.x.begin(), R.x.end(), P.x.begin(), P.x.begin(), func);
    // thrust::transform(R.y.begin(), R.y.end(), P.y.begin(), P.y.begin(), func);
    // thrust::transform(R.z.begin(), R.z.end(), P.z.begin(), P.z.begin(), func);
    // thrust::transform(R.xy.begin(), R.xy.end(), P.xy.begin(), P.xy.begin(), func);
    // thrust::transform(R.yz.begin(), R.yz.end(), P.yz.begin(), P.yz.begin(), func);
    // thrust::transform(R.xz.begin(), R.xz.end(), P.xz.begin(), P.xz.begin(), func);

    // cudaDeviceSynchronize();

    /// Applying Absorbing Boundary Condition
    // thrust::transform(U.x.begin(), U.x.end(), B.begin(), U.x.begin(), thrust::multiplies<float>());
    // thrust::transform(U.y.begin(), U.y.end(), B.begin(), U.y.begin(), thrust::multiplies<float>());
    // thrust::transform(U.z.begin(), U.z.end(), B.begin(), U.z.begin(), thrust::multiplies<float>());
    // thrust::transform(P.x.begin(), P.x.end(), B.begin(), P.x.begin(), thrust::multiplies<float>());
    // thrust::transform(P.y.begin(), P.y.end(), B.begin(), P.y.begin(), thrust::multiplies<float>());
    // thrust::transform(P.z.begin(), P.z.end(), B.begin(), P.z.begin(), thrust::multiplies<float>());
    // thrust::transform(P.xy.begin(), P.xy.end(), B.begin(), P.xy.begin(), thrust::multiplies<float>());
    // thrust::transform(P.yz.begin(), P.yz.end(), B.begin(), P.yz.begin(), thrust::multiplies<float>());
    // thrust::transform(P.xz.begin(), P.xz.end(), B.begin(), P.xz.begin(), thrust::multiplies<float>());
    // thrust::transform(R.x.begin(), R.x.end(), B.begin(), R.x.begin(), thrust::multiplies<float>());
    // thrust::transform(R.y.begin(), R.y.end(), B.begin(), R.y.begin(), thrust::multiplies<float>());
    // thrust::transform(R.z.begin(), R.z.end(), B.begin(), R.z.begin(), thrust::multiplies<float>());
    // thrust::transform(R.xy.begin(), R.xy.end(), B.begin(), R.xy.begin(), thrust::multiplies<float>());
    // thrust::transform(R.yz.begin(), R.yz.end(), B.begin(), R.yz.begin(), thrust::multiplies<float>());
    // thrust::transform(R.xz.begin(), R.xz.end(), B.begin(), R.xz.begin(), thrust::multiplies<float>());

    // cudaDeviceSynchronize();

    m_i += 1;
};

template<unsigned int x, unsigned int y, unsigned int z, unsigned int N, typename T>
void Scene<x, y, z, N, T>::AddMaterial(Material m) {
    m_materials.push_back(m);
}

template<unsigned int x, unsigned int y, unsigned int z, unsigned int N, typename T>
void Scene<x, y, z, N, T>::PrintMaterials() const {
    std::cout << "Materials : ";
    for (auto const &m : m_materials)
        std::cout << m << " ";
}

template<unsigned int x, unsigned int y, unsigned int z, unsigned int N, typename T>
void Scene<x, y, z, N, T>::SetScene(thrust::device_vector<unsigned int> &M) {
    // Checking scene size
    if (M.size() != m_d.x * m_d.y * m_d.z)
        throw std::invalid_argument("Scene vector size does not match the Scene dimensions !");

    // Checking undefined material
    // unsigned int n = m_materials.size();
    // bool undefined_material = thrust::transform_reduce(M.begin(), M.end(), CheckUndefinedMaterial(n), false, thrust::plus<bool>());
    // if (undefined_material)
    //     throw std::invalid_argument("Scene vector contains uninitialized materials !");

    m_M = M;
}

template<unsigned int x, unsigned int y, unsigned int z, unsigned int N, typename T>
void Scene<x, y, z, N, T>::TriggerNextEvent() {
    Event e = m_events.top();
    if (m_i < e.i())
        throw std::invalid_argument("Scene time is prior to the time of the next Event !");

    if (m_i > e.i())
        throw std::invalid_argument("Scene time is later to the time of the next Event !");

    e.Callback();
    m_events.pop();
}

template<unsigned int x, unsigned int y, unsigned int z, unsigned int N, typename T>
void Scene<x, y, z, N, T>::Init() {
    m_alpha[0] = 1.f / (24.f * m_dx.x);
    m_alpha[1] = 1.f / (24.f * m_dx.y);
    m_alpha[2] = 1.f / (24.f * m_dx.z);

    /// Allocating DeviceMaterial
    FrequencyDomain fd = m_frequency_domain;
    for (const auto &m : m_materials) {
        m_device_materials.push_back(m.GetDeviceMaterial<float>(fd));
    }

    std::cout << "Eta Tau P : [";
    thrust::copy(m_device_materials.eta_tau_p.begin(), m_device_materials.eta_tau_p.end(), std::ostream_iterator<float>(std::cout, " "));
    std::cout << "]\n";
    std::cout << "Eta Tau Gamma P : [";
    thrust::copy(m_device_materials.eta_tau_gamma_p.begin(), m_device_materials.eta_tau_gamma_p.end(), std::ostream_iterator<float>(std::cout, " "));
    std::cout << "]\n";
    std::cout << "Mu Tau S : [";
    thrust::copy(m_device_materials.mu_tau_s.begin(), m_device_materials.mu_tau_s.end(), std::ostream_iterator<float>(std::cout, " "));
    std::cout << "]\n";
    std::cout << "Mu Tau Gamma S : [";
    thrust::copy(m_device_materials.mu_tau_gamma_s.begin(), m_device_materials.mu_tau_gamma_s.end(), std::ostream_iterator<float>(std::cout, " "));
    std::cout << "]\n";

    /// Allocating Tau Sigma
    std::vector<float> tau_sigma = m_frequency_domain.TauSigma();
    m_tau_sigma = thrust::device_vector<float>(tau_sigma.begin(), tau_sigma.end());
}