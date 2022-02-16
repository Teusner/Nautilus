#pragma once

#include <string>
#include <vector>

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

inline void to_xarray(std::string filename, std::vector<float> &Px, std::vector<float> &Py, std::vector<float> &Pz, const std::size_t x, const std::size_t y, const std::size_t z) {
    const std::array<std::size_t, 3> shape = {x, y, z};
    auto x_Px = xt::adapt(Px, shape);
    auto x_Py = xt::adapt(Py, shape);
    auto x_Pz = xt::adapt(Pz, shape);
    xt::xtensor<float, 4> array = xt::stack(xt::xtuple(x_Px, x_Py, x_Pz), 3);
    xt::dump_npy(filename, array);
}

template<std::size_t x, std::size_t y, std::size_t z, typename ...Args>
void dump_numpy(std::string filename, Args ...args) {
    std::cout << filename << std::endl;
    const std::array<std::size_t, 3> shape = {x, y, z};
    xt::xtensor<float, 4> array = xt::stack(xt::xtuple(xt::adapt(args..., shape)), 3);
    xt::dump_npy(filename, array);
}