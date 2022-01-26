#pragma once

#include <queue>
#include <vector>

#include "Material.cuh"
#include "Field.cuh"
#include "Event.h"
#include "Module.cuh"


class Scene {
    /// Constructor with a dimension
    public: Scene(const unsigned int x, const unsigned int y, const unsigned int z, const float dx, const float dy, const float dz, const float dt);

    /// Scene Dimension Getter
    public: dim3 Dims() const { return m_d; };

    /// Add a material to the Scene
    public: void AddMaterial(Material m);

    /// Print material in the scene
    public: void PrintMaterials() const;

    /// Copy materials to constant memory
    public: void AllocateMaterials(const void* symbol) const;

    /// Scene Matrix Getter
    public: thrust::device_vector<unsigned int> GetScene() const { return m_M; };

    /// Scene Matrix Setter
    public: void SetScene(thrust::device_vector<unsigned int> &M);

    /// Simulation Time
    /// Return the current simulation time
    public: double Time() const { return double(m_dt * m_i); };

    /// Time Step Getter
    public: float TimeStep() const { return m_dt; };

    /// X-Step Getter
    public: float X() const { return m_d.x; };

    /// Y-Step Getter
    public: float Y() const { return m_d.y; };

    /// Z-Step Getter
    public: float Z() const { return m_d.z; };

    /// X-Step Getter
    public: float dX() const { return m_dx.x; };

    /// Y-Step Getter
    public: float dY() const { return m_dx.y; };

    /// Z-Step Getter
    public: float dZ() const { return m_dx.z; };

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
    public: unsigned int m_i;
    
    /// Scene dimension
    private: const dim3 m_d;

    /// X-step
    private: const float3 m_dx;

    /// Priority Queue
    /// Priority queue handling Events in priority order
    private : std::priority_queue<Event, std::vector<Event>, std::greater<Event>> m_events;

    /// Vector of Material
    private: std::vector<Material> m_materials;

    /// Vector of Emmitters
    public: thrust::device_vector<Emitter> emitters;

    /// Scene description vector
    private: thrust::device_vector<unsigned int> m_M;

    /// Pressure Field
    public: PressureField P;

    /// Velocity Field
    public: VelocityField U;

    /// Memory Field
    public: MemoryField R;

    /// Emitter Pressure Field
    public: thrust::device_vector<float> F;
};
