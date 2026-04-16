#pragma once

#include <torch/extension.h>

#define WINGS_ASCEND_CHECK(...) TORCH_CHECK(__VA_ARGS__)
