#pragma once
#include <cmath>

#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/detail/config/host_device.h>
#include <iostream>

#include "FrequencyDomain.cuh"

template<typename T>
struct DeviceMaterial {
    T inv_rho;
    T eta_tau_p;
    T eta_tau_gamma_p;
    T mu_tau_s;
    T mu_tau_gamma_s;
};

template<typename Vector>
struct DeviceMaterials {

    /// Type Definitions
    typedef typename Vector::value_type T;
    typedef thrust::zip_iterator<
        thrust::tuple<
            typename Vector::iterator,
            typename Vector::iterator,
            typename Vector::iterator,
            typename Vector::iterator,
            typename Vector::iterator
        >
    > iterator;

    /// Constructor
    DeviceMaterials(std::size_t size_) : size(size_), inv_rho(size), eta_tau_p(size), eta_tau_gamma_p(size), mu_tau_s(size), mu_tau_gamma_s(size) {};

    /// Member variables
    std::size_t size;
    Vector inv_rho;
    Vector eta_tau_p;
    Vector eta_tau_gamma_p;
    Vector mu_tau_s;
    Vector mu_tau_gamma_s;

    /// Copy operator
    template <typename TOther>
    DeviceMaterials<Vector>& operator=(const TOther &other) {
        inv_rho = other.inv_rho;
        eta_tau_p = other.eta_tau_p;
        eta_tau_gamma_p = other.eta_tau_gamma_p;
        mu_tau_s = other.mu_tau_s;
        mu_tau_gamma_s = other.mu_tau_gamma_s;
        return *this;
    }

    /// Begin iterator
    iterator begin() {
        return thrust::make_zip_iterator(thrust::make_tuple(inv_rho.begin(),eta_tau_p.begin(),eta_tau_gamma_p.begin(),mu_tau_s.begin(),mu_tau_gamma_s.begin()));
    }

    /// End iterator
    iterator end() {
        return thrust::make_zip_iterator(thrust::make_tuple(inv_rho.end(),eta_tau_p.end(),eta_tau_gamma_p.end(),mu_tau_s.end(),mu_tau_gamma_s.end()));
    }

    /// Array of structure getter at index
    struct Ref {
        T &inv_rho; T &eta_tau_p; T &eta_tau_gamma_p; T &mu_tau_s; T &mu_tau_gamma_s;
        Ref(iterator z) : inv_rho(thrust::get<0>(z)), eta_tau_p(thrust::get<1>(z)), eta_tau_gamma_p(thrust::get<2>(z)), mu_tau_s(thrust::get<3>(z)), mu_tau_gamma_s(thrust::get<4>(z)) {}
    };

    void push_back(DeviceMaterial<T> dm) {
        inv_rho.push_back(dm.inv_rho);
        eta_tau_p.push_back(dm.eta_tau_p);
        eta_tau_gamma_p.push_back(dm.eta_tau_gamma_p);
        mu_tau_s.push_back(dm.mu_tau_s);
        mu_tau_gamma_s.push_back(dm.mu_tau_gamma_s);
    };
};


class Material {
    public:
        __host__ Material() : Material(1000, 1500, 100, 0, 2) {};
        __host__ Material(float rho, float cp, float Qp, float cs, float Qs) : m_rho(rho), m_cp(cp), m_Qp(Qp), m_cs(cs), m_Qs(Qs) {};
        __host__ ~Material() = default;

        __host__ float Rho() const { return m_rho; };
        __host__ float Cp() const { return m_cp; };
        __host__ float Qp() const { return m_Qp; };
        __host__ float Cs() const { return m_cs; };
        __host__ float Qs() const { return m_Qs; };

        __host__ void CopyToConstant(const void* symbol, unsigned int index) const;

        template<typename T>
        __host__ DeviceMaterial<T> GetDeviceMaterial(FrequencyDomain fd) const;

    private:
        /// Qp and Qs could be set for each SLS in a Material
        /// Qp and Qs become std::vector of size L
        float m_rho;
        float m_cp;
        float m_Qp;
        float m_cs = 0.f;
        float m_Qs = 0.f;
};

std::ostream &operator<<(std::ostream &os, const Material &m);
std::ostream &operator<<(std::ostream &os, const Material *m);


/// Implementation
template<typename T>
DeviceMaterial<T> Material::GetDeviceMaterial(FrequencyDomain fd) const {
    /// Tau epsilon p computing
    float tau_gamma_p = fd.tau(m_Qp);
    float tau_p = tau_gamma_p + 1;

    /// Tau epsilon s computing
    float tau_gamma_s = fd.tau(m_Qs);
    float tau_s = tau_gamma_s + 1;

    std::cout << "Tau P : " << tau_p << std::endl;
    std::cout << "Tau S : " << tau_s << std::endl;

    std::cout << "Eta : " << m_rho * powf(m_cp, 2) << std::endl;
    std::cout << "Mu : " << m_rho * powf(m_cs, 2) << std::endl;

    return DeviceMaterial<float>{1 / m_rho, m_rho * powf(m_cp, 2) * tau_p, m_rho * powf(m_cp, 2) * tau_gamma_p, m_rho * powf(m_cs, 2) * tau_s, m_rho * powf(m_cs, 2) * tau_gamma_s};
}

inline std::ostream &operator<<(std::ostream &os, const Material &m) {
    return os << "{rho: " << m.Rho() << ", P: ["
                        << m.Cp() << ", " << m.Qp() << "], S: ["
                        << m.Cs() << ", " << m.Qs() << "]}";
}

inline std::ostream &operator<<(std::ostream &os, const Material *m) {
    return os << "{rho: " << m->Rho() << ", P: ["
                        << m->Cp() << ", " << m->Qp() << "], S: ["
                        << m->Cs() << ", " << m->Qs() << "]}";
}

template<typename T>
inline std::ostream &operator<<(std::ostream &os, const DeviceMaterial<T> &m) {
    return os << "{1/rho: " << m.inv_rho << ", Eta/P : [tau_p :"
                        << m.eta_tau_p << ", tau_gamma_p : " << m.eta_tau_gamma_p << "], Mu/S: [tau_s : "
                        << m.mu_tau_s << ", tau_gamma_s : " << m.mu_tau_gamma_s << "]}";
}

template<typename T>
inline std::ostream &operator<<(std::ostream &os, const DeviceMaterial<T> *m) {
    return os << "{rho: " << m->inv_rho << ", P : ["
                        << m->eta_tau_p << ", " << m->eta_tau_gamma_p << "], S: ["
                        << m->mu_tau_s << ", " << m->mu_tau_gamma_s << "]}";
}