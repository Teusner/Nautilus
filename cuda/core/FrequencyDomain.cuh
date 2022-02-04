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
    public: FrequencyDomain(const float omega_min, const float omega_max, unsigned int l_ = 3);

    /// Constructor by giving tau_sigma
    public: FrequencyDomain(const float omega_min, const float omega_max, std::vector<float> tau_sigma);

    /// Omega Min Getter
    public: float OmegaMin() const { return m_min; };

    /// Omega Max Getter
    public: float OmegaMax() const { return m_max; };

    /// Numbre of relaxation constraints getter
    public: unsigned int l() const { return m_l; };

    /// Tau Sigma Getter
    public: std::vector<float> TauSigma() const { return m_tau_sigma; };

    /// Optimal tau computation over omega range by giving Q_0
    public: float tau(float Q_0) const { return m_I / Q_0; };

    /// Pulsation range
    private: float m_min;
    private: float m_max;

    /// Number of relaxation constraints
    private: unsigned int m_l;

    /// Relaxation constraints
    private: std::vector<float> m_tau_sigma;

    /// I computation
    private: void I();

    /// I = Q_0 * tau = cste, used to compute optimal tau for each Material
    private: float m_I;
};

/// Implementation
inline FrequencyDomain::FrequencyDomain(const float omega_min, const float omega_max, unsigned int l_) {
    m_l = l_;
    m_min = std::min(omega_min, omega_max);
    m_max = std::max(omega_min, omega_max);

    if (m_min == m_max)
        throw std::invalid_argument("Omega range is null !");

    // Tau distribution over omega range
    int i = -1;
    auto f = [&] () {
        i++;
        if (i==0) return 1 / m_max;
        if (i==m_l-1) return 1 / m_min;
        return (1 / m_min + 1 / m_max) / float(std::pow(2., m_l-i-1));
    };
    m_tau_sigma.resize(m_l);
    if (m_l == 1) {
        m_tau_sigma[0] = 2. / (m_min + m_max);
    }
    else {
        std::generate(std::begin(m_tau_sigma), std::end(m_tau_sigma), f);
    }

    /// Compute m_I
    this->I();
}

inline FrequencyDomain::FrequencyDomain(const float omega_min, const float omega_max, std::vector<float> tau_sigma) {
    m_min = std::min(omega_min, omega_max);
    m_max = std::max(omega_min, omega_max);

    if (m_min == m_max)
        throw std::invalid_argument("Omega range is null !");
    if (tau_sigma.size() == 0)
        throw std::invalid_argument("Tau sigma is empty !");

    m_l = tau_sigma.size();
    m_tau_sigma = tau_sigma;

    /// Compute m_I
    this->I();
}

inline void FrequencyDomain::I() {
    // I computing
    std::vector<float> I0l;
    std::transform(std::begin(m_tau_sigma), std::end(m_tau_sigma), std::back_inserter(I0l), [&](float t_s) { return (std::log((1 + std::pow(m_max*t_s, 2)) / (1 + std::pow(m_min*t_s, 2)))) / (2*t_s); });

    std::vector<float> I1l;
    std::transform(std::begin(m_tau_sigma), std::end(m_tau_sigma), std::back_inserter(I1l), [&](float t_s) { return (std::atan(m_max*t_s) - std::atan(m_min*t_s) + m_min*t_s/(1+std::pow(m_min*t_s, 2)) - m_max*t_s/(1+std::pow(m_max*t_s, 2))) / (2*t_s); });

    unsigned int l = 0;
    unsigned int k = 1;
    auto f21 = [&] (float w) { 
        return std::atan(w*m_tau_sigma[l])/m_tau_sigma[l] - std::atan(w*m_tau_sigma[k])/m_tau_sigma[k];
    };
    auto I2kl = [&]() {
        float val =  m_tau_sigma[l]*m_tau_sigma[k] / (std::pow(m_tau_sigma[k], 2) - std::pow(m_tau_sigma[l], 2)) * (f21(m_max) - f21(m_min));
        if (k == m_l-1) { l++; k = l; }
        k++;
        return val;
    };

    std::vector<float> I2l(m_l*(m_l-1)/2);
    std::generate(std::begin(I2l), std::end(I2l), I2kl);

    m_I = std::accumulate(std::begin(I0l), std::end(I0l), 0.) / (std::accumulate(std::begin(I1l), std::end(I1l), 0.) + 2 * std::accumulate(std::begin(I2l), std::end(I2l), 0.));
    std::cout << "I : " << m_I << std::endl;
}