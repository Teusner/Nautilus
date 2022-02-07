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
    T eta_tau_p_1;
    T mu_tau_s_1;
};

template<typename Vector>
struct DeviceMaterials {

    /// Type Definitions
    typedef typename Vector::value_type T;
    typedef thrust::zip_iterator<
        thrust::tuple<
            typename Vector::iterator,
            typename Vector::iterator,
            typename Vector::iterator
        >
    > iterator;

    /// Constructor
    DeviceMaterials(std::size_t size_) : size(size_), inv_rho(size), eta_tau_p_1(size), mu_tau_s_1(size) {};

    /// Member variables
    std::size_t size;
    Vector inv_rho;
    Vector eta_tau_p_1;
    Vector mu_tau_s_1;

    /// Copy operator
    template <typename TOther>
    DeviceMaterials<Vector>& operator=(const TOther &other) {
        inv_rho = other.inv_rho;
        eta_tau_p_1 = other.eta_tau_p_1;
        mu_tau_s_1 = other.mu_tau_s_1;
        return *this;
    }

    /// Begin iterator
    iterator begin() {
        return thrust::make_zip_iterator(thrust::make_tuple(inv_rho.begin(),eta_tau_p_1.begin(),mu_tau_s_1.begin()));
    }

    /// End iterator
    iterator end() {
        return thrust::make_zip_iterator(thrust::make_tuple(inv_rho.end(),eta_tau_p_1.end(),mu_tau_s_1.end()));
    }

    /// Array of structure getter at index
    struct Ref {
        T &inv_rho; T &eta_tau_p; T &eta_tau_p_1; T &mu_tau_s; T &mu_tau_s_1;
        Ref(iterator z) : inv_rho(thrust::get<0>(z)), eta_tau_p_1(thrust::get<1>(z)), mu_tau_s_1(thrust::get<2>(z)) {}
    };

    void push_back(DeviceMaterial<T> dm) {
        inv_rho.push_back(dm.inv_rho);
        eta_tau_p_1.push_back(dm.eta_tau_p_1);
        mu_tau_s_1.push_back(dm.mu_tau_s_1);
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

// /// Really necessary ?
// inline void Material::CopyToConstant(const void* symbol, unsigned int index) const {
//     // Copying one material on constant memory
//     DeviceMaterial *temp_h_m = (DeviceMaterial*) malloc(sizeof(DeviceMaterial) * N);
//     cudaMemcpyFromSymbol(temp_h_m, symbol, sizeof(DeviceMaterial)*N);

//     // Filling the i-th DeviceMaterial
//     temp_h_m[index].inv_rho = 1 / m_rho;
//     temp_h_m[index].eta_tau_gamma_p = 1.2 - 1;
//     temp_h_m[index].mu_tau_gamma_s = 1.2 - 1;

//     cudaMemcpyToSymbol(symbol, temp_h_m, sizeof(DeviceMaterial)*N);

//     free(temp_h_m);
// }
template<typename T>
DeviceMaterial<T> Material::GetDeviceMaterial(FrequencyDomain fd) const {
    /// Getting l
    unsigned int l = fd.l();

    /// Getting Tau Sigma
    std::vector<float> tau_sigma = fd.TauSigma();

    /// Tau Sigma showing
    std::cout << "L: " << l << std::endl;
    std::cout << "Tau Sigma: [";
    std::copy(std::begin(tau_sigma), std::end(tau_sigma), std::ostream_iterator<float>(std::cout, ", "));
    std::cout << "]\n";

    /// Tau epsilon p computing
    float tau_p = fd.tau(m_Qp);
    std::cout << "Tau_p = " << tau_p << std::endl;

    /// Tau epsilon s computing
    float tau_s = fd.tau(m_Qs);
    std::cout << "Tau_s = " << tau_s << std::endl;

    return DeviceMaterial<float>{1 / m_rho, m_rho * powf(m_cp, 2.) * (tau_p + 1), m_rho * powf(m_cs, 2) * (tau_s + 1)};
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