#pragma once

#include <algorithm>
#include <vector>

class FrequencyDomain {
    /// Default Constructor
    public: FrequencyDomain() {};

    /// Constructor with omega range
    public: FrequencyDomain(const float omega_min, const float omega_max);

    /// Omega Min Getter
    public: float OmegaMin() const { return m_min; };

    /// Omega Max Getter
    public: float OmegaMax() const { return m_max; };

    /// Optimal tau computation over omega range by giving Q_0
    public: float tau(float Q_0) const;

    /// Pulsation range
    private: float m_min;
    private: float m_max;
    private: std::vector<float> tau_sigma;
};

/// Implementation
inline FrequencyDomain::FrequencyDomain(const float omega_min, const float omega_max) {
    m_min = std::min(omega_min, omega_max);
    m_max = std::max(omega_min, omega_max);
}