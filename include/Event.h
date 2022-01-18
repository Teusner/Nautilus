#pragma once
#include <functional>


class Event {
    public:
        Event() : Event(0) {};
        Event(unsigned int i) : m_i(i) {};
        Event(unsigned int i, std::function<void(void)> f) : m_i(i), m_f(f) {};

        unsigned int i() const { return m_i; };
        std::function<void(void)> Callback() const { return m_f; };

    private:
        /// Time
        unsigned int m_i;

        /// Callable event
        std::function<void(void)> m_f = nullptr;
};

bool operator<(const Event &lhs, const Event &rhs);
bool operator>(const Event &lhs, const Event &rhs);
bool operator<=(const Event &lhs, const Event &rhs);
bool operator>=(const Event &lhs, const Event &rhs);