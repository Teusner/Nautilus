#pragma once
#include <vector>
#include "core/Material.cuh"
#include "core/Module.cuh"

void CopyMaterialToSymbol(const void* symbol, const std::vector<Material> &materials);
