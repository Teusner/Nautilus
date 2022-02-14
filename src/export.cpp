#include "export.h"

#include <string>
#include <vector>

#include <xtensor/xarray.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xnpy.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xio.hpp>

void to_xarray(std::string filename, std::vector<float> &M, std::vector<std::size_t> shape) {
    xt::xarray<float> array = xt::adapt(M, shape);
    xt::dump_npy(filename, array);
};