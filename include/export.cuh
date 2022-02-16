#pragma once

#include <string>
#include <vector>

#include <xtensor/xtensor.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xnpy.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xio.hpp>

void to_one_xarray(std::string filename, std::vector<float> &B, const std::size_t x, const std::size_t y, const std::size_t z);
void to_xarray(std::string filename, std::vector<float> &Px, std::vector<float> &Py, std::vector<float> &Pz, const std::size_t x, const std::size_t y, const std::size_t z);

template<std::size_t x, std::size_t y, std::size_t z, typename ...Args>
void dump_numpy(std::string filename, Args ...args) {
    std::cout << filename << std::endl;
    const std::array<std::size_t, 3> shape = {x, y, z};
    xt::xtensor<float, 4> array = xt::stack(xt::xtuple(xt::adapt(args..., shape)), 3);
    xt::dump_npy(filename, array);
}