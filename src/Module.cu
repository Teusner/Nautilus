# include "Module.h"

#include <iostream>

void Module::print() {
    std::cout << "t : [ ";
    for (auto const &i : m_t)
        std::cout << i << " ";
    std::cout << "]\n";
    std::cout << "s : [ ";
    for (auto const &i : m_s)
        std::cout << i << " ";
    std::cout << "]\n";
}

float Emitter::operator()(float x) {
    float v = m_f(x);
    m_t.push_back(x);
    m_s.push_back(v);
    return v;
}

void Reciever::Record(float t, float s) {
    m_t.push_back(t);
    m_s.push_back(s);
}