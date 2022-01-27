#pragma once

#include <algorithm>
#include <cmath>
#include <numeric>
#include <utility>
#include <vector>

class FrequencyDomain {
    /// Default Constructor
    public: FrequencyDomain() {};

    /// Constructor with omega range and the number of relaxation variables
    public: FrequencyDomain(const float omega_min, const float omega_max, unsigned int n = 5);

    /// Omega Min Getter
    public: float OmegaMin() const { return m_min; };

    /// Omega Max Getter
    public: float OmegaMax() const { return m_max; };

    /// Tau Sigma Getter
    public: std::vector<float> TauSigma() const { return tau_sigma; };

    /// Optimal tau computation over omega range by giving Q_0
    public: float tau(float Q_0) const;

    /// Pulsation range
    private: float m_min;
    private: float m_max;

    /// Relaxation constraints
    private: std::vector<float> tau_sigma;
};

/// Implementation
inline FrequencyDomain::FrequencyDomain(const float omega_min, const float omega_max, unsigned int n) {
    m_min = std::min(omega_min, omega_max);
    m_max = std::max(omega_min, omega_max);

    if (m_min == m_max)
        throw std::invalid_argument("Omega range is null !");

    unsigned int i = 0;
    tau_sigma.resize(n);
    std::generate(std::begin(tau_sigma), std::end(tau_sigma), [&] () { i++; return std::pow(2., i)/(m_max - m_min); });

    std::cout << "Omega range: [" << m_min << ", " << m_max << "] s\n";
    std::cout << "Tau range: [" << 1 / m_max << ", " << 1 / m_min << "] s\n";
    std::cout << "Tau_sigma distribution : [";
    for (const auto & i : tau_sigma) {
        std::cout << i << ", ";
    }
    std::cout << "]\n";
}