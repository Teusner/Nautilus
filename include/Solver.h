#pragma once

#include <queue>
#include "Simulation.h"


class Solver {
    public: __host__ Solver() {};

    public : std::priority_queue<Event, std::vector<Event>, std::greater<Event>> Events;
};

