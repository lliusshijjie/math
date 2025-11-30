#include <gtest/gtest.h>
#include <math/core/special.hpp>
#include <math/core/constants.hpp>

using namespace math::core;

TEST(SpecialTest, GammaFunctions) {
    // tgamma (using math::core::gamma wrapper)
    EXPECT_NEAR(math::core::gamma(5.0), 24.0, 1e-9);
    EXPECT_NEAR(math::core::gamma(1.0), 1.0, 1e-9);
    EXPECT_NEAR(math::core::gamma(0.5), std::sqrt(PI), 1e-9);

    // lgamma
    EXPECT_NEAR(math::core::lgamma(5.0), std::log(24.0), 1e-9);
    EXPECT_NEAR(math::core::lgamma(1.0), 0.0, 1e-9);
}

TEST(SpecialTest, BetaFunction) {
    // Beta(a, b) = Gamma(a)*Gamma(b)/Gamma(a+b)
    // Beta(2, 3) = 1! * 2! / 4! = 2 / 24 = 1/12
    EXPECT_NEAR(math::core::beta(2.0, 3.0), 1.0/12.0, 1e-9);
    EXPECT_NEAR(math::core::beta(1.0, 1.0), 1.0, 1e-9);
}

TEST(SpecialTest, ErrorFunctions) {
    // erf
    EXPECT_NEAR(math::core::erf(0.0), 0.0, 1e-9);
    // erf(1) approx 0.84270079
    EXPECT_NEAR(math::core::erf(1.0), 0.84270079, 1e-7);
    
    // erfc
    EXPECT_NEAR(math::core::erfc(0.0), 1.0, 1e-9);
    EXPECT_NEAR(math::core::erfc(1.0), 1.0 - 0.84270079, 1e-7);
}

TEST(SpecialTest, RegularizedGamma) {
    // P(1, x) = 1 - exp(-x)
    EXPECT_NEAR(regularized_gamma_p(1.0, 1.0), 1.0 - std::exp(-1.0), 1e-7);
    EXPECT_NEAR(regularized_gamma_p(1.0, 0.0), 0.0, 1e-9);
    
    // Check series vs CF crossover region (usually x ~ a+1)
    // a=2, x=1 (series)
    // P(2, 1) = 1 - (1+1)e^-1 = 1 - 2/e
    EXPECT_NEAR(regularized_gamma_p(2.0, 1.0), 1.0 - 2.0/E, 1e-7);
    
    // a=1, x=5 (CF)
    EXPECT_NEAR(regularized_gamma_p(1.0, 5.0), 1.0 - std::exp(-5.0), 1e-7);
}

TEST(SpecialTest, IncompleteBeta) {
    // I_x(1, 1) = x
    EXPECT_NEAR(incomplete_beta(1.0, 1.0, 0.5), 0.5, 1e-9);
    EXPECT_NEAR(incomplete_beta(1.0, 1.0, 0.2), 0.2, 1e-9);
    
    // Symmetry I_x(a, b) = 1 - I_{1-x}(b, a)
    double val1 = incomplete_beta(2.0, 3.0, 0.4);
    double val2 = 1.0 - incomplete_beta(3.0, 2.0, 0.6);
    EXPECT_NEAR(val1, val2, 1e-9);
    
    // Known values
    // I_0.5(a, a) = 0.5
    EXPECT_NEAR(incomplete_beta(3.0, 3.0, 0.5), 0.5, 1e-9);
}
