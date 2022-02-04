#pragma once
#include "Material.cuh"
#define N 10

__constant__ DeviceMaterial M[N];
__constant__ unsigned int d_l;
__constant__ float d_dt;
__constant__ float d_tau_sigma[N];
__constant__ float d_alpha[3];