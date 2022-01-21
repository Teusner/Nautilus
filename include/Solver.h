#pragma once

#include "Event.h"
#include "Scene.h"

#include <queue>
#include <vector>


class Solver {
    /// Default Constructor
    public: Solver(const Scene &s) : m_s(s) {};

    /// Priority Queue
    /// Priority queue handling Events in priority order
    public : std::priority_queue<Event, std::vector<Event>, std::greater<Event>> Events;

    /// Step Simulation
    /// Compute the next state
    public : void Step();

    /// Run Simulation
    /// Run the simulation until the next event
    /// in the priority queue
    public: void RunNext();

    /// Simulation Time
    /// Return the current simulation time
    public: double Time() const { return double(m_h * m_i); };

    /// Time Increment
    /// Return the current time increment
    public: unsigned int Increment() const { return m_i; };

    /// Time Step
    private: double m_h;
    
    /// Time Increment
    private: unsigned int m_i;

    /// Scene
    private: Scene m_s;
};
