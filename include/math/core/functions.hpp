#pragma once

#include <algorithm>
#include <type_traits>

namespace math::core {

/**
 * @brief Clamps a value between a minimum and a maximum.
 * 
 * @tparam T Value type
 * @param v Value to clamp
 * @param lo Minimum value
 * @param hi Maximum value
 * @return Reference to v clamped between lo and hi
 */
template <typename T>
[[nodiscard]] constexpr const T& clamp(const T& v, const T& lo, const T& hi) {
    return std::clamp(v, lo, hi);
}

/**
 * @brief Linear interpolation between a and b.
 *        Returns a + t * (b - a).
 * 
 * @tparam T Value type
 * @tparam U Interpolation factor type
 * @param a Start value
 * @param b End value
 * @param t Interpolation factor
 * @return Interpolated value
 */
template <typename T, typename U>
[[nodiscard]] constexpr T lerp(T a, T b, U t) {
    static_assert(std::is_floating_point_v<U>, "Interpolation factor must be a floating point type");
    return a + static_cast<T>(t) * (b - a);
}

} // namespace math::core
