#pragma once

#include "Scene.cuh"
#include "kernels.cuh"


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

template<unsigned int x, unsigned int y, unsigned int z>
void Solver::Step(Scene &s) const {
    dim3 ThreadPerBlock(4, 4, 4);
    dim3 GridDimension(x / ThreadPerBlock.x, y / ThreadPerBlock.y, z / ThreadPerBlock.z);

    Ux<x, y, z><<<GridDimension, ThreadPerBlock>>>(
        s.TimeStep(),
        thrust::raw_pointer_cast(&(s.U.x[0])),
        thrust::raw_pointer_cast(&(s.P.x[0])),
        thrust::raw_pointer_cast(&(s.P.xy[0])),
        thrust::raw_pointer_cast(&(s.P.xz[0])),
        thrust::raw_pointer_cast(&(s.GetScene()[0]))
    );

    Uy<x, y, z><<<GridDimension, ThreadPerBlock>>>(
        s.TimeStep(),
        thrust::raw_pointer_cast(&(s.U.x[0])),
        thrust::raw_pointer_cast(&(s.P.x[0])),
        thrust::raw_pointer_cast(&(s.P.xy[0])),
        thrust::raw_pointer_cast(&(s.P.xz[0])),
        thrust::raw_pointer_cast(&(s.GetScene()[0]))
    );

    Uz<x, y, z><<<GridDimension, ThreadPerBlock>>>(
        s.TimeStep(),
        thrust::raw_pointer_cast(&(s.U.x[0])),
        thrust::raw_pointer_cast(&(s.P.x[0])),
        thrust::raw_pointer_cast(&(s.P.xy[0])),
        thrust::raw_pointer_cast(&(s.P.xz[0])),
        thrust::raw_pointer_cast(&(s.GetScene()[0]))
    );
};
