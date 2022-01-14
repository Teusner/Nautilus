#pragma once

#include <queue>
#include "Event.h"


class Solver {
    public: Solver() {};

    public : std::priority_queue<Event, std::vector<Event>, std::greater<Event>> Events;
};

