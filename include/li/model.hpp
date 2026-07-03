#pragma once

#include <cstdint>
#include "li/types.hpp"
#include "li/status.hpp"

namespace li {

struct Model {
    double alpha;
    double beta; 
};

struct Moments {
    uint64_t n;
    double sum_x;
    double sum_y;
    double sum_xx;
    double sum_xy;
};

// minimizes residual error from moments
[[nodiscard]] inline Model least_squares(Moments m) {
    LI_ASSERT(m.n > 0);

    // n is converted to double here with causes precision loss,
    // but realistically segment count will not get that high

    // variance of x
    double d = static_cast<double>(m.n) * m.sum_xx - m.sum_x * m.sum_x;

    // n == 1, no unique slope
    if (d == 0) {
        return Model{ 0.0, m.sum_y / static_cast<double>(m.n) };
    }

    double alpha = (m.n * m.sum_xy - m.sum_x * m.sum_y) / d;
    double beta = (m.sum_y - alpha * m.sum_x) / static_cast<double>(m.n);
    return Model{ alpha, beta };
}

}
