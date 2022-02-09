#include "export.h"

#include <algorithm>
#include <string>
#include <vector>

#include <xtensor/xarray.hpp>
#include <xtensor/xnpy.hpp>

void to_xarray(std::string filename, std::vector<float>::iterator begin, std::vector<float>::iterator end, std::size_t x, std::size_t y, std::size_t z) {
    xt::xarray<float>::shape_type shape = {x, y, z};
    xt::xarray<float> M(shape);
    std::copy(begin, end, M.begin());
    xt::dump_npy(filename, M);
};