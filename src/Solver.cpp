#include "Solver.h"
#include "Event.h"

void Solver::Step() {
    /// Implement step
}

void Solver::RunNext() {
    Event e = Events.top();
    while (e.i() < m_i) {
        Step();
    }
    Events.pop();
}