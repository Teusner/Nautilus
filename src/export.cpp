#include "export.h"

#include <string>
#include <vector>

#include <xtensor/xtensor.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xnpy.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xio.hpp>

void to_one_xarray(std::string filename, std::vector<float> &B, const std::size_t x, const std::size_t y, const std::size_t z) {
    const std::array<std::size_t, 3> shape = {x, y, z};
    xt::xtensor<float, 3> array = xt::adapt(B, shape);
    xt::dump_npy(filename, array);
};

void to_xarray(std::string filename, std::vector<float> &Px, std::vector<float> &Py, std::vector<float> &Pz, const std::size_t x, const std::size_t y, const std::size_t z) {
    const std::array<std::size_t, 3> shape = {x, y, z};
    auto x_Px = xt::adapt(Px, shape);
    auto x_Py = xt::adapt(Py, shape);
    auto x_Pz = xt::adapt(Pz, shape);
    xt::xtensor<float, 4> array = xt::stack(xt::xtuple(x_Px, x_Py, x_Pz), 3);
    xt::dump_npy(filename, array);
};
