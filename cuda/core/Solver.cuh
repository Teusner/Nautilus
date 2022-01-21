#pragma once

#include "Scene.cuh"

class Solver {
    /// Default Constructor
    public: Solver() {};

    /// Step Simulation
    /// Compute the next state
    public: template<unsigned int x, unsigned int y, unsigned int z> void Step(Scene &s) const;

    /// Run Simulation
    /// Run the simulation until the next event
    /// in the priority queue of the scene
    public: void RunNext(Scene &s) const;
};
