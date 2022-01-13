#pragma once

#include <functional>


class Event {
    public:
        __host__ Event() : Event(0) {};
        __host__ Event(unsigned int i) : m_i(i) {};
        __host__ Event(unsigned int i, std::function<void(void)> f) : m_i(i), m_f(f) {};

        __host__ unsigned int i() const {return m_i;};
        __host__ std::function<void(void)> Callback() const {return m_f;};

    private:
        /// Time
        unsigned int m_i;

        /// Callable event
        std::function<void(void)> m_f = nullptr;
};

bool operator< (const Event &e1, const Event &e2);