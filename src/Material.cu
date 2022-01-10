#include "Material.h"
#include <stdio.h>

__global__ void NewMaterial(float rho, float cp, float qp, Material* m){
    *m = *(new Material(rho, cp, qp));
}

__global__ void PrintMaterial(Material* m) {
    printf("{rho: %d, P: [%d, %d], S: [%d, %d]}", m->Rho(), m->Cp(), m->Qp(), m->Cs(), m->Qs());
}

std::ostream &operator<<(std::ostream &os, const Material& m) {
    return os << "{rho: " << m.Rho() << ", P: ["
                        << m.Cp() << ", " << m.Qp() << "], S: ["
                        << m.Cs() << ", " << m.Qs() << "]}";
}