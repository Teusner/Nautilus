#pragma once

#include <string>
#include <vector>

void to_xarray(std::string filename, std::vector<float> &Px, std::vector<float> &Py, std::vector<float> &Pz, const std::size_t x, const std::size_t y, const std::size_t z);
