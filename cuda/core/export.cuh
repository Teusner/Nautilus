#pragma once

#include <string>
#include <vector>

#include <xtensor/xarray.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xnpy.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xio.hpp>

inline void to_one_xarray(std::string filename, std::vector<float> &B, const std::size_t x, const std::size_t y, const std::size_t z){
    const std::array<std::size_t, 3> shape = {x, y, z};
    xt::xtensor<float, 3> array = xt::adapt(B, shape);
    xt::dump_npy(filename, array);
}

template<unsigned int x, unsigned int y, unsigned int z>
inline void to_xarray(std::string filename, std::vector<float> &Px, std::vector<float> &Py, std::vector<float> &Pz) {
    xt::xtensor<float, 3>::shape_type shape = {x, y, z};
    xt::xtensor<float, 3> x_Px = xt::adapt(Px, shape);
    xt::xtensor<float, 3> x_Py = xt::adapt(Py, shape);
    xt::xtensor<float, 3> x_Pz = xt::adapt(Pz, shape);
    xt::dump_npy(filename, xt::stack(xt::xtuple(x_Px, x_Py, x_Pz), 3));
}

template<std::size_t x, std::size_t y, std::size_t z, typename ...Args>
void dump_numpy(std::string filename, Args ...args) {
    std::cout << filename << std::endl;
    const std::array<std::size_t, 3> shape = {x, y, z};
    xt::xtensor<float, 4> array = xt::stack(xt::xtuple(xt::adapt(args..., shape)), 3);
    xt::dump_npy(filename, array);
}