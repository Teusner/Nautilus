#pragma once
#include <thrust/device_vector.h>
#include <thrust/iterator/zip_iterator.h>

template<typename Vector>
struct VelocityField {

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
    VelocityField(std::size_t size_) : size(size_), x(size), y(size), z(size) {};

    /// Member variables
    std::size_t size;
    Vector x;
    Vector y;
    Vector z;

    /// Copy operator
    template <typename TOther>
    VelocityField<Vector>& operator=(const TOther &other) {
        x = other.x;
        y = other.y;
        z = other.z;
        return *this;
    }

    /// Begin iterator
    iterator begin() {
        return thrust::make_zip_iterator(thrust::make_tuple(x.begin(),y.begin(),z.begin()));
    }

    /// End iterator
    iterator end() {
        return thrust::make_zip_iterator(thrust::make_tuple(x.end(),y.end(),z.end()));
    }

    /// Array of structure getter at index
    struct Ref {
        T &x; T &y; T &z;
        Ref(iterator z) : x(thrust::get<0>(z)), y(thrust::get<1>(z)), z(thrust::get<2>(z)) {}
    };
};

template<typename Vector>
struct PressureField {

    /// Type Definitions
    typedef typename Vector::value_type T;
    typedef thrust::zip_iterator<
        thrust::tuple<
            typename Vector::iterator,
            typename Vector::iterator,
            typename Vector::iterator,
            typename Vector::iterator,
            typename Vector::iterator,
            typename Vector::iterator
        >
    > iterator;

    /// Constructor
    PressureField(std::size_t size_) : size(size_), x(size), y(size), z(size), xy(size), yz(size), xz(size) {};

    /// Member variables
    std::size_t size;
    Vector x;
    Vector y;
    Vector z;
    Vector xy;
    Vector yz;
    Vector xz;

    /// Copy operator
    template <typename TOther>
    PressureField<Vector>& operator=(const TOther &other) {
        x = other.x;
        y = other.y;
        z = other.z;
        xy = other.xy;
        yz = other.yz;
        xz = other.xz;
        return *this;
    }

    /// Begin iterator
    iterator begin() {
        return thrust::make_zip_iterator(thrust::make_tuple(x.begin(),y.begin(),z.begin(),xy.begin(),yz.begin(),xz.begin()));
    }

    /// End iterator
    iterator end() {
        return thrust::make_zip_iterator(thrust::make_tuple(x.end(),y.end(),z.end(),xy.end(),yz.end(),xz.end()));
    }

    /// Array of structure getter at index
    struct Ref {
        T &x; T &y; T &z; T &xy; T &yz; T &xz;
        Ref(iterator z) : x(thrust::get<0>(z)), y(thrust::get<1>(z)), z(thrust::get<2>(z)), xy(thrust::get<3>(z)), yz(thrust::get<4>(z)), yz(thrust::get<5>(z)){}
    };
};

template<typename Vector>
struct MemoryField {

    /// Type Definitions
    typedef typename Vector::value_type T;
    typedef thrust::zip_iterator<
        thrust::tuple<
            typename Vector::iterator,
            typename Vector::iterator,
            typename Vector::iterator,
            typename Vector::iterator,
            typename Vector::iterator,
            typename Vector::iterator
        >
    > iterator;

    /// Constructor
    MemoryField(std::size_t size_) : size(size_), x(size), y(size), z(size), xy(size), yz(size), xz(size) {};

    /// Member variables
    std::size_t size;
    Vector x;
    Vector y;
    Vector z;
    Vector xy;
    Vector yz;
    Vector xz;

    /// Copy operator
    template <typename TOther>
    MemoryField<Vector>& operator=(const TOther &other) {
        x = other.x;
        y = other.y;
        z = other.z;
        xy = other.xy;
        yz = other.yz;
        xz = other.xz;
        return *this;
    }

    /// Begin iterator
    iterator begin() {
        return thrust::make_zip_iterator(thrust::make_tuple(x.begin(),y.begin(),z.begin(),xy.begin(),yz.begin(),xz.begin()));
    }

    /// End iterator
    iterator end() {
        return thrust::make_zip_iterator(thrust::make_tuple(x.end(),y.end(),z.end(),xy.end(),yz.end(),xz.end()));
    }

    /// Array of structure getter at index
    struct Ref {
        T &x; T &y; T &z; T &xy; T &yz; T &xz;
        Ref(iterator z) : x(thrust::get<0>(z)), y(thrust::get<1>(z)), z(thrust::get<2>(z)), xy(thrust::get<3>(z)), yz(thrust::get<4>(z)), yz(thrust::get<5>(z)){}
    };
};
