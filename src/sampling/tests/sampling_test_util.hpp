// Shared helpers for sampling module tests (doctest).
#pragma once

#include <doctest/doctest.h>

#include <cmath>

#define CHECK_NEAR(a, b, eps) CHECK(std::fabs(static_cast<double>(a) - static_cast<double>(b)) <= (eps))
