#include "Solver.cuh"
#include "Scene.cuh"
#include "Event.h"
#include "kernels.cuh"

#include <thrust/device_vector.h>


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
}

void Solver::RunNext(Scene &s) const {
    // Event e = Events.top();
    // while (e.i() < m_i) {
    //     Step();
    // }
    // Events.pop();
}