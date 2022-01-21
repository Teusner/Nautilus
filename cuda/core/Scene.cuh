#pragma once
#include <queue>
#include <vector>

#include "core/Material.cuh"
#include "core/Field.cuh"
#include "Event.h"


class Scene {
    /// Constructor with a dimension
    public: Scene(dim3 d, float dx, float dy, float dz, float dt);

    /// Copy Constructor
    public: Scene(const Scene& s) : m_d(s.Dims()), m_dx(s.XStep()), m_dy(s.YStep()), m_dz(s.ZStep()), P(s.Dims()), U(s.Dims()), R(s.Dims()){};

    /// Move constructor
    public: Scene(Scene&&) = default;

    /// Scene Dimension Getter
    public: dim3 Dims() const { return m_d; };

    /// Add a material to the Scene
    public: void AddMaterial(Material m);

    /// Print material in the scene
    public: void PrintMaterials() const;

    /// Copy materials to constant memory
    public: void AllocateMaterials(const void* symbol) const;

    /// Scene Matrix Getter
    public: thrust::device_vector<float> GetScene() const { return m_M; };

    /// Scene Matrix Setter
    public: void SetScene(thrust::device_vector<float> M);

    /// Simulation Time
    /// Return the current simulation time
    public: double Time() const { return double(m_dt * m_i); };

    /// Time Step Getter
    public: float TimeStep() const { return m_dt; };

    /// X-Step Getter
    public: float XStep() const { return m_dx; };

    /// Y-Step Getter
    public: float YStep() const { return m_dy; };

    /// Z-Step Getter
    public: float ZStep() const { return m_dz; };

    /// Time Increment
    /// Return the current time increment
    public: unsigned int Increment() const { return m_i; };

    /// Next Event Getter
    public: Event GetNextEvent() const { return m_events.top(); };

    /// Next Event Trigger
    public: void TriggerNextEvent();

    /// Time Step
    private: double m_dt;
    
    /// Time Increment
    private: unsigned int m_i;
    
    /// Scene dimension
    private: const dim3 m_d;

    /// X-step
    private: const float m_dx;

    /// Y-step
    private: const float m_dy;

    /// Z-step
    private: const float m_dz;

    /// Priority Queue
    /// Priority queue handling Events in priority order
    private : std::priority_queue<Event, std::vector<Event>, std::greater<Event>> m_events;

    /// Vector of Material
    private: std::vector<Material> m_materials;

    /// Scene description vector
    private: thrust::device_vector<float> m_M;

    /// Pressure Field
    public: PressureField P;

    /// Velocity Field
    public: VelocityField U;

    /// Memory Field
    public: MemoryField R;
};