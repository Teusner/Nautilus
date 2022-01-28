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
    public: float tau(float Q_0) const { return m_I / Q_0; };

    /// Pulsation range
    private: float m_min;
    private: float m_max;

    /// Relaxation constraints
    private: std::vector<float> tau_sigma;

    /// I = Q_0 * tau = cste, used to compute optimal tau for each Material
    private: float m_I;
};

/// Implementation
inline FrequencyDomain::FrequencyDomain(const float omega_min, const float omega_max, unsigned int n) {
    m_min = std::min(omega_min, omega_max);
    m_max = std::max(omega_min, omega_max);

    if (m_min == m_max)
        throw std::invalid_argument("Omega range is null !");

    // Tau distribution over omega range
    int i = -1;
    auto f = [&] () {
        i++;
        if (i==0) return 1/m_max;
        if (i==n-1) return 1/m_min;
        return (1/m_min + 1/m_max) / float(std::pow(2., n-i-1));
    };
    tau_sigma.resize(n);
    std::generate(std::begin(tau_sigma), std::end(tau_sigma), f);

    // I computing
    std::vector<float> I0l;
    std::transform(std::begin(tau_sigma), std::end(tau_sigma), std::back_inserter(I0l), [&](float t_s) { return (std::log((1 + std::pow(m_max*t_s, 2)) / (1 + std::pow(m_min*t_s, 2)))) / (2*t_s); });

    std::vector<float> I1l;
    std::transform(std::begin(tau_sigma), std::end(tau_sigma), std::back_inserter(I1l), [&](float t_s) { return (std::atan(m_max*t_s) - std::atan(m_min*t_s) + m_min*t_s/(1+std::pow(m_min*t_s, 2)) - m_max*t_s/(1+std::pow(m_max*t_s, 2))) / (2*t_s); });

    unsigned int l = 0;
    unsigned int k = 1;
    auto f21 = [&] (float w) { 
        return std::atan(w*tau_sigma[l])/tau_sigma[l] - std::atan(w*tau_sigma[k])/tau_sigma[k];
    };
    auto I2kl = [&]() {
        float val =  tau_sigma[l]*tau_sigma[k] / (std::pow(tau_sigma[k], 2) - std::pow(tau_sigma[l], 2)) * (f21(m_max) - f21(m_min));
        if (k == n-1) { l++; k = l; }
        k++;
        return val;
    };

    std::vector<float> I2l(n*(n-1)/2);
    std::generate(std::begin(I2l), std::end(I2l), I2kl);

    m_I = std::accumulate(std::begin(I0l), std::end(I0l), 0.) / (std::accumulate(std::begin(I1l), std::end(I1l), 0.) + 2 * std::accumulate(std::begin(I2l), std::end(I2l), 0.));
}