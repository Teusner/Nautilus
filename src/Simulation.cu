#include "Simulation.h"

bool operator< (const Event &e1, const Event &e2) {
    return e1.i() < e2.i();
}