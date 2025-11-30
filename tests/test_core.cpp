#include <gtest/gtest.h>
#include <math/core/constants.hpp>
#include <math/core/utils.hpp>
#include <math/core/functions.hpp>

using namespace math::core;

TEST(CoreTest, Constants) {
    EXPECT_NEAR(PI, 3.1415926535, 1e-9);
    EXPECT_NEAR(TWO_PI, 6.2831853071, 1e-9);
    EXPECT_NEAR(HALF_PI, 1.5707963267, 1e-9);
    EXPECT_NEAR(E, 2.7182818284, 1e-9);
}

TEST(CoreTest, UtilsEquals) {
    EXPECT_TRUE(equals(1.0, 1.0));
    // Default epsilon is very small (~2.22e-15 for double), so 1e-10 difference is too big
    EXPECT_FALSE(equals(1.0, 1.0 + 1e-10)); 
    
    // Test with custom epsilon
    EXPECT_TRUE(equals(1.0, 1.0 + 1e-10, 1e-9));
    EXPECT_FALSE(equals(1.0, 1.1, 0.01));
}

TEST(CoreTest, UtilsAngleConversion) {
    EXPECT_TRUE(equals(radians(180.0), PI));
    EXPECT_TRUE(equals(radians(90.0), HALF_PI));
    EXPECT_TRUE(equals(degrees(PI), 180.0));
    EXPECT_TRUE(equals(degrees(HALF_PI), 90.0));
}

TEST(CoreTest, FunctionsClamp) {
    EXPECT_EQ(clamp(5, 0, 10), 5);
    EXPECT_EQ(clamp(-5, 0, 10), 0);
    EXPECT_EQ(clamp(15, 0, 10), 10);
    
    EXPECT_DOUBLE_EQ(clamp(0.5, 0.0, 1.0), 0.5);
    EXPECT_DOUBLE_EQ(clamp(-0.1, 0.0, 1.0), 0.0);
    EXPECT_DOUBLE_EQ(clamp(1.1, 0.0, 1.0), 1.0);
}

TEST(CoreTest, FunctionsLerp) {
    EXPECT_DOUBLE_EQ(lerp(0.0, 10.0, 0.5), 5.0);
    EXPECT_DOUBLE_EQ(lerp(0.0, 10.0, 0.0), 0.0);
    EXPECT_DOUBLE_EQ(lerp(0.0, 10.0, 1.0), 10.0);
    EXPECT_DOUBLE_EQ(lerp(0.0, 10.0, 0.25), 2.5);
}
