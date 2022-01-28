#pragma once

#include "kernels.cuh"
#include "Scene.cuh"


class Solver {
    /// Default Constructor
    public: Solver() {};

    public: void Init(Scene &s) const;

    /// Step Simulation
    /// Compute the next state
    public: template<unsigned int x, unsigned int y, unsigned int z, typename T> void Step(Scene &s) const;

    /// Run Simulation
    /// Run the simulation until the next event
    /// in the priority queue of the scene
    public: void RunNext(Scene &s) const;
};

inline void Solver::Init(Scene &s) const {
    float dt = s.TimeStep();
    cudaMemcpyToSymbol(dt, &dt, sizeof(float), 0, cudaMemcpyHostToDevice);
    unsigned int l = s.l();
    cudaMemcpyToSymbol(l, &l, sizeof(unsigned int), 0, cudaMemcpyHostToDevice);
}

template<unsigned int x, unsigned int y, unsigned int z, typename T>
void Solver::Step(Scene &s) const {

    // Emitter Field computing
    F<x, y, z, T><<<1, 1>>>(s.Time(), thrust::raw_pointer_cast(&(s.emitters[0])), thrust::raw_pointer_cast(&(s.F[0])));

    float tau_sigma = 0.1;
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

    Rxx<x, y, z><<<GridDimension, ThreadPerBlock>>>(
        s.TimeStep(),
        thrust::raw_pointer_cast(&(s.R.x[0])),
        thrust::raw_pointer_cast(&(s.U.x[0])),
        thrust::raw_pointer_cast(&(s.U.y[0])),
        thrust::raw_pointer_cast(&(s.U.z[0])),
        thrust::raw_pointer_cast(&(s.GetScene()[0])),
        tau_sigma
    );

    Ryy<x, y, z><<<GridDimension, ThreadPerBlock>>>(
        s.TimeStep(),
        thrust::raw_pointer_cast(&(s.R.y[0])),
        thrust::raw_pointer_cast(&(s.U.x[0])),
        thrust::raw_pointer_cast(&(s.U.y[0])),
        thrust::raw_pointer_cast(&(s.U.z[0])),
        thrust::raw_pointer_cast(&(s.GetScene()[0])),
        tau_sigma
    );

    Rzz<x, y, z><<<GridDimension, ThreadPerBlock>>>(
        s.TimeStep(),
        thrust::raw_pointer_cast(&(s.R.z[0])),
        thrust::raw_pointer_cast(&(s.U.x[0])),
        thrust::raw_pointer_cast(&(s.U.y[0])),
        thrust::raw_pointer_cast(&(s.U.z[0])),
        thrust::raw_pointer_cast(&(s.GetScene()[0])),
        tau_sigma
    );

    Rxy<x, y, z><<<GridDimension, ThreadPerBlock>>>(
        s.TimeStep(),
        thrust::raw_pointer_cast(&(s.R.xy[0])),
        thrust::raw_pointer_cast(&(s.U.x[0])),
        thrust::raw_pointer_cast(&(s.U.y[0])),
        thrust::raw_pointer_cast(&(s.GetScene()[0])),
        tau_sigma
    );

    Ryz<x, y, z><<<GridDimension, ThreadPerBlock>>>(
        s.TimeStep(),
        thrust::raw_pointer_cast(&(s.R.yz[0])),
        thrust::raw_pointer_cast(&(s.U.y[0])),
        thrust::raw_pointer_cast(&(s.U.z[0])),
        thrust::raw_pointer_cast(&(s.GetScene()[0])),
        tau_sigma
    );

    Rxz<x, y, z><<<GridDimension, ThreadPerBlock>>>(
        s.TimeStep(),
        thrust::raw_pointer_cast(&(s.R.xz[0])),
        thrust::raw_pointer_cast(&(s.U.x[0])),
        thrust::raw_pointer_cast(&(s.U.z[0])),
        thrust::raw_pointer_cast(&(s.GetScene()[0])),
        tau_sigma
    );

    Pxx<x, y, z><<<GridDimension, ThreadPerBlock>>>(
        s.TimeStep(),
        thrust::raw_pointer_cast(&(s.P.x[0])),
        thrust::raw_pointer_cast(&(s.U.x[0])),
        thrust::raw_pointer_cast(&(s.U.y[0])),
        thrust::raw_pointer_cast(&(s.U.z[0])),
        thrust::raw_pointer_cast(&(s.R.x[0])),
        thrust::raw_pointer_cast(&(s.GetScene()[0])),
        thrust::raw_pointer_cast(&(s.F[0]))
    );

    Pyy<x, y, z><<<GridDimension, ThreadPerBlock>>>(
        s.TimeStep(),
        thrust::raw_pointer_cast(&(s.P.y[0])),
        thrust::raw_pointer_cast(&(s.U.x[0])),
        thrust::raw_pointer_cast(&(s.U.y[0])),
        thrust::raw_pointer_cast(&(s.U.z[0])),
        thrust::raw_pointer_cast(&(s.R.y[0])),
        thrust::raw_pointer_cast(&(s.GetScene()[0])),
        thrust::raw_pointer_cast(&(s.F[0]))
    );

    Pzz<x, y, z><<<GridDimension, ThreadPerBlock>>>(
        s.TimeStep(),
        thrust::raw_pointer_cast(&(s.P.z[0])),
        thrust::raw_pointer_cast(&(s.U.x[0])),
        thrust::raw_pointer_cast(&(s.U.y[0])),
        thrust::raw_pointer_cast(&(s.U.z[0])),
        thrust::raw_pointer_cast(&(s.R.z[0])),
        thrust::raw_pointer_cast(&(s.GetScene()[0])),
        thrust::raw_pointer_cast(&(s.F[0]))
    );

    Pxy<x, y, z><<<GridDimension, ThreadPerBlock>>>(
        s.TimeStep(),
        thrust::raw_pointer_cast(&(s.P.xy[0])),
        thrust::raw_pointer_cast(&(s.U.x[0])),
        thrust::raw_pointer_cast(&(s.U.y[0])),
        thrust::raw_pointer_cast(&(s.R.xy[0])),
        thrust::raw_pointer_cast(&(s.GetScene()[0]))
    );

    Pyz<x, y, z><<<GridDimension, ThreadPerBlock>>>(
        s.TimeStep(),
        thrust::raw_pointer_cast(&(s.P.yz[0])),
        thrust::raw_pointer_cast(&(s.U.y[0])),
        thrust::raw_pointer_cast(&(s.U.z[0])),
        thrust::raw_pointer_cast(&(s.R.yz[0])),
        thrust::raw_pointer_cast(&(s.GetScene()[0]))
    );

    Pxz<x, y, z><<<GridDimension, ThreadPerBlock>>>(
        s.TimeStep(),
        thrust::raw_pointer_cast(&(s.P.xz[0])),
        thrust::raw_pointer_cast(&(s.U.x[0])),
        thrust::raw_pointer_cast(&(s.U.z[0])),
        thrust::raw_pointer_cast(&(s.R.xz[0])),
        thrust::raw_pointer_cast(&(s.GetScene()[0]))
    );
};
