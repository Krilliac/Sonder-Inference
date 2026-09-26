// Shared helpers for sampling module tests (doctest).
#pragma once

// Include full iostream declarations before doctest: MSVC 14.5x otherwise sees
// doctest's forward-declared std::basic_ostream while parsing <string_view>.
#include <ostream>
#include <string>
#include <string_view>

#include <doctest/doctest.h>

#include <cmath>

#define CHECK_NEAR(a, b, eps) CHECK(std::fabs(static_cast<double>(a) - static_cast<double>(b)) <= (eps))
