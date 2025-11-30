#pragma once

#include <cmath>
#include <limits>
#include <type_traits>
#include "math/core/constants.hpp"

namespace math::core {

/**
 * @brief Compare two floating point numbers for equality within a given epsilon.
 * 
 * @tparam T Floating point type
 * @param a First number
 * @param b Second number
 * @param epsilon Tolerance (default: machine epsilon * 10)
 * @return true if |a - b| <= epsilon
 */
template <typename T>
[[nodiscard]] constexpr bool equals(T a, T b, T epsilon = std::numeric_limits<T>::epsilon() * 10) {
    static_assert(std::is_floating_point_v<T>, "T must be a floating point type");
    return std::abs(a - b) <= epsilon;
}

/**
 * @brief Convert degrees to radians.
 * 
 * @tparam T Floating point type
 * @param deg Angle in degrees
 * @return Angle in radians
 */
template <typename T>
[[nodiscard]] constexpr T radians(T deg) {
    static_assert(std::is_floating_point_v<T>, "T must be a floating point type");
    return deg * static_cast<T>(PI / 180.0);
}

/**
 * @brief Convert radians to degrees.
 * 
 * @tparam T Floating point type
 * @param rad Angle in radians
 * @return Angle in degrees
 */
template <typename T>
[[nodiscard]] constexpr T degrees(T rad) {
    static_assert(std::is_floating_point_v<T>, "T must be a floating point type");
    return rad * static_cast<T>(180.0 / PI);
}

} // namespace math::core
